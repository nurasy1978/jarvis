import { createSdkMcpServer, tool } from '@anthropic-ai/claude-agent-sdk'
import { z } from 'zod'
import { PROXY_UA } from './net.mjs'

/**
 * JARVIS on YouTube.
 *
 * Three skills, all reads: find videos, describe one, and read what is said in
 * it. That last one is the reason this exists. "Summarise this video", "what
 * did he say about the battery", "skip to the part about the reactor" are all
 * questions about the words in a video, and a model that can only see a title
 * and a thumbnail answers them by guessing. With the transcript it answers them
 * the way a person who had watched it would — and with a timestamp, so the
 * blade can open the player at the right second.
 *
 * No API key. Everything comes from the same public pages a browser loads:
 * the search results page and the watch page both carry their data as a JSON
 * blob in the HTML, and the caption tracks they list can be fetched directly.
 * That is less stable than a documented API — YouTube reshapes these blobs
 * every so often — so every parser here is a walk that looks for the shape it
 * wants wherever it happens to be, rather than a fixed path that breaks the day
 * a wrapper object is added.
 *
 * Why not through the media proxy's SSRF gate: every URL here is built by this
 * file against a fixed YouTube host, and the one URL that comes back from
 * YouTube (a caption track) is checked against that host before it is fetched.
 * The model never chooses where a request goes — it only supplies a search
 * string or an eleven-character video id.
 */

const ORIGIN = 'https://www.youtube.com'
const TIMEOUT_MS = 12_000
const MAX_PAGE_BYTES = 6 * 1024 * 1024

/** A transcript longer than this is cut, with a note saying how to page it. */
const MAX_TRANSCRIPT_CHARS = 60_000

/**
 * Without these, a request from an EU address is redirected to the consent
 * interstitial and every parse below finds nothing. SOCS=CAI is the "rejected
 * all non-essential cookies" answer, which is also the one we would give.
 */
const HEADERS = {
  'user-agent': PROXY_UA,
  'accept-language': 'en-GB,en;q=0.9',
  cookie: 'SOCS=CAI; CONSENT=YES+cb',
}

/** The `sp` filter YouTube's own UI sends: videos only, with the sort folded in. */
const SORT_FILTER = {
  relevance: 'EgIQAQ%3D%3D',
  recent: 'CAISAhAB',
  views: 'CAMSAhAB',
}

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

/**
 * The video id out of anything a person or a search result might hand over:
 * a bare id, watch?v=, youtu.be/, /shorts/, /embed/, /live/, music. and m.
 * Returns null rather than guessing — a wrong id plays the wrong video.
 */
export function videoIdOf(raw) {
  const s = String(raw ?? '').trim()
  if (/^[\w-]{11}$/.test(s)) return s
  let url
  try {
    url = new URL(/^https?:\/\//i.test(s) ? s : `https://${s}`)
  } catch {
    return null
  }
  const host = url.hostname.toLowerCase().replace(/^(?:www|m|music)\./, '')
  let id = ''
  if (host === 'youtu.be') id = url.pathname.split('/')[1] ?? ''
  else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    if (url.pathname === '/watch') id = url.searchParams.get('v') ?? ''
    else {
      const m = /^\/(?:shorts|embed|live|v)\/([\w-]{11})/.exec(url.pathname)
      id = m?.[1] ?? ''
    }
  }
  return /^[\w-]{11}$/.test(id) ? id : null
}

/** "1:02:03" and "4:05" to seconds; null when it is neither. */
export function clockToSeconds(clock) {
  const parts = String(clock ?? '').trim().split(':')
  if (!parts.length || parts.some((p) => !/^\d+$/.test(p))) return null
  return parts.reduce((total, p) => total * 60 + Number(p), 0)
}

/** Seconds to "m:ss" or "h:mm:ss", the way YouTube itself writes it. */
export function secondsToClock(total) {
  const t = Math.max(0, Math.floor(Number(total) || 0))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = String(t % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

async function get(url, accept = 'text/html,*/*;q=0.8') {
  const res = await fetch(url, {
    headers: { ...HEADERS, accept },
    redirect: 'follow',
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (!res.ok) throw new Error(`YouTube answered ${res.status}`)
  const declared = Number(res.headers.get('content-length'))
  if (Number.isFinite(declared) && declared > MAX_PAGE_BYTES) throw new Error('response too large')
  const text = await res.text()
  if (text.length > MAX_PAGE_BYTES) throw new Error('response too large')
  return text
}

/**
 * The object literal that follows `marker` in a page, parsed.
 *
 * A regex up to `};` is the usual shortcut and it is wrong often enough to
 * matter: video descriptions are full of braces and semicolons. This walks the
 * braces, skipping over string contents, so it stops at the real end.
 */
export function extractJson(html, marker) {
  const at = html.indexOf(marker)
  if (at < 0) return null
  const start = html.indexOf('{', at + marker.length)
  if (start < 0) return null
  let depth = 0
  let inString = false
  for (let i = start; i < html.length; i++) {
    const c = html[i]
    if (inString) {
      if (c === '\\') i++
      else if (c === '"') inString = false
      continue
    }
    if (c === '"') inString = true
    else if (c === '{') depth++
    else if (c === '}' && --depth === 0) {
      try {
        return JSON.parse(html.slice(start, i + 1))
      } catch {
        return null
      }
    }
  }
  return null
}

/** Every object under `root` that has the key `key`, depth first, in order. */
function* findAll(root, key) {
  const stack = [root]
  while (stack.length) {
    const node = stack.pop()
    if (!node || typeof node !== 'object') continue
    if (!Array.isArray(node) && Object.hasOwn(node, key)) yield node[key]
    const children = Array.isArray(node) ? node : Object.values(node)
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i])
  }
}

/** YouTube writes text as either {simpleText} or {runs: [{text}]}. */
export function textOf(t) {
  if (!t) return ''
  if (typeof t === 'string') return t
  if (typeof t.simpleText === 'string') return t.simpleText
  if (Array.isArray(t.runs)) return t.runs.map((r) => r?.text ?? '').join('')
  if (typeof t.content === 'string') return t.content
  return ''
}

const bestThumb = (thumbs) => {
  const list = Array.isArray(thumbs) ? thumbs : []
  const url = list.at(-1)?.url ?? ''
  return url.startsWith('//') ? `https:${url}` : url.split('?')[0]
}

const watchUrl = (id, seconds) =>
  `${ORIGIN}/watch?v=${id}${seconds ? `&t=${Math.floor(seconds)}s` : ''}`

const fail = (text) => ({ isError: true, content: [{ type: 'text', text }] })
const ok = (text) => ({ content: [{ type: 'text', text }] })

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/** The video results out of a search page's ytInitialData. */
export function parseSearch(data, limit) {
  const out = []
  const seen = new Set()
  for (const v of findAll(data, 'videoRenderer')) {
    if (!v?.videoId || seen.has(v.videoId)) continue
    seen.add(v.videoId)
    const live = (v.badges ?? []).some((b) =>
      /LIVE/i.test(b?.metadataBadgeRenderer?.style ?? b?.metadataBadgeRenderer?.label ?? ''),
    )
    out.push({
      id: v.videoId,
      title: textOf(v.title),
      channel: textOf(v.ownerText ?? v.longBylineText ?? v.shortBylineText),
      duration: textOf(v.lengthText) || (live ? 'LIVE' : ''),
      views: textOf(v.viewCountText) || textOf(v.shortViewCountText),
      published: textOf(v.publishedTimeText),
      snippet: (v.detailedMetadataSnippets ?? [])
        .map((s) => textOf(s?.snippetText))
        .join(' ')
        .slice(0, 240),
      thumbnail: bestThumb(v.thumbnail?.thumbnails),
      url: watchUrl(v.videoId),
    })
    if (out.length >= limit) break
  }
  return out
}

const SEARCH_DESCRIPTION = `Search YouTube for videos.

Returns up to a dozen results, each with its title, channel, length, view count,
age, a thumbnail and the watch URL. Use it whenever the user asks for a video —
"find me a video on", "show me the trailer", "what has this channel posted",
"play some" — rather than a general web search, which returns pages about videos
instead of the videos.

Then act on what they asked for:
  - to watch one, open it on a blade with kind "embed" and its URL;
  - to choose between several, put the thumbnails up as a gallery and say which
    one you would pick, in a sentence;
  - to know what is IN a video, call youtube_transcript — never summarise a
    video from its title.

Never read the list aloud. Name one or two, by title, at most.`

// ---------------------------------------------------------------------------
// The watch page: details and caption tracks
// ---------------------------------------------------------------------------

/** Chapters, from the player bar's markers or, failing that, the description. */
export function parseChapters(initialData, description) {
  const out = []
  for (const c of findAll(initialData ?? {}, 'chapterRenderer')) {
    const at = Number(c?.timeRangeStartMillis)
    if (!Number.isFinite(at)) continue
    out.push({ at: Math.round(at / 1000), title: textOf(c.title) })
  }
  if (out.length) return dedupeChapters(out)

  // The same chapters, written by hand in the description: lines that start
  // with a timestamp. YouTube only honours them when the first is 0:00, so the
  // same rule is applied here — otherwise any description that mentions a time
  // would sprout chapters.
  for (const line of String(description ?? '').split('\n')) {
    const m = /^\s*[([]?((?:\d{1,2}:)?\d{1,2}:\d{2})[)\]]?\s*[-–—:|]?\s*(.+)$/.exec(line)
    if (m) out.push({ at: clockToSeconds(m[1]), title: m[2].trim() })
  }
  return out.length >= 2 && out[0].at === 0 ? out : []
}

const dedupeChapters = (list) => {
  const seen = new Set()
  return list.filter((c) => (seen.has(c.at) ? false : (seen.add(c.at), true)))
}

/** The parts of a player response JARVIS has any use for. */
export function parseDetails(player, initialData) {
  const d = player?.videoDetails ?? {}
  const micro = player?.microformat?.playerMicroformatRenderer ?? {}
  const description = d.shortDescription ?? textOf(micro.description) ?? ''
  const tracks = player?.captions?.playerCaptionsTracklistRenderer?.captionTracks ?? []
  return {
    id: d.videoId,
    title: d.title ?? textOf(micro.title),
    channel: d.author ?? micro.ownerChannelName ?? '',
    seconds: Number(d.lengthSeconds) || null,
    views: Number(d.viewCount) || null,
    published: micro.publishDate ?? micro.uploadDate ?? '',
    live: Boolean(d.isLiveContent),
    category: micro.category ?? '',
    keywords: (d.keywords ?? []).slice(0, 12),
    description,
    chapters: parseChapters(initialData, description),
    captions: tracks.map((t) => ({
      lang: t.languageCode,
      name: textOf(t.name),
      auto: t.kind === 'asr',
      baseUrl: t.baseUrl,
    })),
    playable: player?.playabilityStatus?.status ?? 'UNKNOWN',
    reason: player?.playabilityStatus?.reason ?? '',
  }
}

async function watchPage(id) {
  const html = await get(`${watchUrl(id)}&hl=en&has_verified=1`)
  return {
    player: extractJson(html, 'ytInitialPlayerResponse = '),
    initialData: extractJson(html, 'ytInitialData = '),
  }
}

/**
 * Caption tracks from the Android client's view of the player.
 *
 * The web watch page increasingly lists caption URLs that return an empty body
 * unless the request carries a proof-of-origin token that only a real browser
 * session can mint. The mobile client's tracks do not need one, so when the web
 * track comes back empty this is the second place to ask. Best effort: the
 * client version is one YouTube accepts today, and may stop accepting.
 */
async function androidTracks(id) {
  const res = await fetch(`${ORIGIN}/youtubei/v1/player?prettyPrint=false`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'user-agent': 'com.google.android.youtube/20.10.38 (Linux; U; Android 14) gzip',
      'accept-language': HEADERS['accept-language'],
    },
    body: JSON.stringify({
      videoId: id,
      context: {
        client: {
          clientName: 'ANDROID',
          clientVersion: '20.10.38',
          androidSdkVersion: 34,
          hl: 'en',
          gl: 'US',
        },
      },
    }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (!res.ok) return []
  return parseDetails(await res.json(), null).captions
}

/**
 * The track to read: the language asked for, else English, else whatever the
 * video was captioned in. A human-written track beats the speech-recognised
 * one in the same language every time.
 */
export function pickTrack(tracks, lang) {
  if (!tracks.length) return null
  const want = String(lang ?? '').toLowerCase().split('-')[0]
  const rank = (t) => {
    const l = String(t.lang ?? '').toLowerCase().split('-')[0]
    return (want && l === want ? 0 : l === 'en' ? 2 : 4) + (t.auto ? 1 : 0)
  }
  return [...tracks].sort((a, b) => rank(a) - rank(b))[0]
}

/** Is this a caption URL we are willing to fetch? Same host, same endpoint. */
const trustedTrackUrl = (raw) => {
  try {
    const url = new URL(raw, ORIGIN)
    return url.protocol === 'https:' &&
      url.hostname === 'www.youtube.com' &&
      url.pathname === '/api/timedtext'
      ? url
      : null
  } catch {
    return null
  }
}

const decodeEntities = (s) =>
  s
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')

/**
 * Caption cues, as [{at, text}], from either format the endpoint speaks.
 * json3 is asked for; the XML form is what comes back from some tracks anyway.
 */
export function parseCaptions(body) {
  const raw = String(body ?? '').trim()
  if (!raw) return []
  if (raw.startsWith('{')) {
    let json
    try {
      json = JSON.parse(raw)
    } catch {
      return []
    }
    return (json.events ?? [])
      .filter((e) => Array.isArray(e.segs))
      .map((e) => ({
        at: (Number(e.tStartMs) || 0) / 1000,
        text: e.segs.map((s) => s.utf8 ?? '').join('').replace(/\s+/g, ' ').trim(),
      }))
      .filter((c) => c.text)
  }
  const cues = []
  // srv1: <text start="1.2" dur="3">…</text>. srv3: <p t="1200" d="3000">…</p>.
  const re = /<(text|p)\b([^>]*)>([\s\S]*?)<\/\1>/g
  for (let m; (m = re.exec(raw)); ) {
    const attrs = m[2]
    const start = /\bstart="([\d.]+)"/.exec(attrs)
    const t = /\bt="(\d+)"/.exec(attrs)
    const at = start ? Number(start[1]) : t ? Number(t[1]) / 1000 : 0
    const text = decodeEntities(decodeEntities(m[3].replace(/<[^>]+>/g, '')))
      .replace(/\s+/g, ' ')
      .trim()
    if (text) cues.push({ at, text })
  }
  return cues
}

async function fetchCues(track) {
  const url = trustedTrackUrl(track?.baseUrl)
  if (!url) return []
  url.searchParams.set('fmt', 'json3')
  return parseCaptions(await get(url.href, 'application/json,text/xml,*/*;q=0.5'))
}

/**
 * Cues grouped into lines of roughly `every` seconds, each stamped once.
 *
 * Raw cues are two or three words long and arrive several per second. Stamping
 * each one triples the token count and buries the words; stamping every twenty
 * seconds or so keeps the transcript readable while still letting the model
 * point at the moment something was said.
 */
export function groupCues(cues, every = 20) {
  const lines = []
  let cur = null
  for (const c of cues) {
    if (!cur || c.at - cur.at >= every) {
      cur = { at: c.at, words: [] }
      lines.push(cur)
    }
    // Auto-captions repeat the tail of the previous cue at the head of the
    // next; drop an exact repeat so the transcript does not stutter.
    if (cur.words.at(-1) !== c.text) cur.words.push(c.text)
  }
  return lines.map((l) => `[${secondsToClock(l.at)}] ${l.words.join(' ')}`)
}

const DETAILS_DESCRIPTION = `Look up one YouTube video: title, channel, length,
views, upload date, description, chapters, and which caption languages exist.

Use it when the user asks about a specific video — "how long is it", "who made
it", "when was it posted", "what are the chapters" — or before reading a
transcript of a long video, so you can choose which chapter to read rather than
reading all of it.`

const TRANSCRIPT_DESCRIPTION = `Read what is said in a YouTube video.

Returns the transcript as lines stamped with the time they were said, in the
video's own captions — human-written where they exist, automatic otherwise.
This is how you answer anything about the content of a video: summarise it,
what did they say about X, does it cover Y, where is the bit about Z.

Rules:
  - Never summarise a video from its title or description. Read this first.
  - Quote the moment, not the minute: when you point at a place in the video,
    give its timestamp and offer to play from there — open it on a blade with
    kind "embed" and the url this tool returns for that moment.
  - Long videos are cut at a limit; the result says so. Use \`from\` and \`to\`
    (seconds) to read a particular stretch, chosen from the chapters.
  - Speak the summary in a sentence or two. If they want detail, put it on a
    blade — never read a transcript aloud.

Some videos have no captions at all (music, very new uploads, some live
streams). Say so in one sentence and offer what the description says instead.`

export function youtubeServer() {
  return createSdkMcpServer({
    name: 'jarvis_youtube',
    version: '1.0.0',
    instructions:
      'YouTube: search for videos, look one up, and read its transcript. Use ' +
      'these rather than a web search for anything to do with a video.',
    // "Find me the video where..." should not depend on tool search noticing
    // that a server exists for it.
    alwaysLoad: true,
    tools: [
      tool(
        'youtube_search',
        SEARCH_DESCRIPTION,
        {
          query: z.string().describe('What to search for, as you would type it into YouTube.'),
          limit: z
            .union([z.number(), z.string()])
            .optional()
            .catch(undefined)
            .describe('How many results, 1 to 12. Default 6.'),
          sort: z
            .enum(['relevance', 'recent', 'views'])
            .optional()
            .catch(undefined)
            .describe('relevance (default), recent = newest uploads first, views = most watched.'),
        },
        async (args) => {
          const q = String(args.query ?? '').trim().slice(0, 200)
          if (!q) return fail('Nothing to search for: `query` was empty.')
          const limit = Math.min(12, Math.max(1, Number(args.limit) || 6))
          const sp = SORT_FILTER[args.sort ?? 'relevance'] ?? SORT_FILTER.relevance
          let results
          try {
            const html = await get(
              `${ORIGIN}/results?search_query=${encodeURIComponent(q)}&sp=${sp}&hl=en`,
            )
            results = parseSearch(extractJson(html, 'ytInitialData = ') ?? {}, limit)
          } catch (err) {
            return fail(`YouTube search failed: ${err?.message ?? err}. Say so in one sentence.`)
          }
          if (!results.length) return ok(`No videos found for "${q}".`)
          return ok(JSON.stringify({ query: q, results }, null, 1))
        },
      ),

      tool(
        'youtube_video',
        DETAILS_DESCRIPTION,
        {
          video: z.string().describe('A YouTube URL of any form, or the eleven-character video id.'),
        },
        async (args) => {
          const id = videoIdOf(args.video)
          if (!id) return fail('That is not a YouTube video link or id.')
          let d
          try {
            const page = await watchPage(id)
            if (!page.player) return fail('YouTube returned a page with no video data on it.')
            d = parseDetails(page.player, page.initialData)
          } catch (err) {
            return fail(`Could not reach YouTube: ${err?.message ?? err}.`)
          }
          if (d.playable !== 'OK' && !d.title) {
            return fail(`That video is unavailable${d.reason ? `: ${d.reason}` : ''}.`)
          }
          return ok(
            JSON.stringify(
              {
                id,
                url: watchUrl(id),
                title: d.title,
                channel: d.channel,
                length: d.seconds ? secondsToClock(d.seconds) : d.live ? 'LIVE' : 'unknown',
                views: d.views,
                published: d.published,
                category: d.category,
                keywords: d.keywords,
                status: d.playable === 'OK' ? 'playable' : `${d.playable} ${d.reason}`.trim(),
                chapters: d.chapters.map((c) => ({
                  at: secondsToClock(c.at),
                  seconds: c.at,
                  title: c.title,
                  url: watchUrl(id, c.at),
                })),
                captions: d.captions.map((c) => `${c.lang}${c.auto ? ' (auto)' : ''}`),
                description: d.description.slice(0, 3000),
              },
              null,
              1,
            ),
          )
        },
      ),

      tool(
        'youtube_transcript',
        TRANSCRIPT_DESCRIPTION,
        {
          video: z.string().describe('A YouTube URL of any form, or the eleven-character video id.'),
          lang: z
            .string()
            .optional()
            .catch(undefined)
            .describe('Preferred caption language code, e.g. "en", "fr". Default: English, else the original.'),
          from: z
            .union([z.number(), z.string()])
            .optional()
            .catch(undefined)
            .describe('Start of the stretch to read, in seconds. Default: the beginning.'),
          to: z
            .union([z.number(), z.string()])
            .optional()
            .catch(undefined)
            .describe('End of the stretch to read, in seconds. Default: the end.'),
        },
        async (args) => {
          const id = videoIdOf(args.video)
          if (!id) return fail('That is not a YouTube video link or id.')

          let details = null
          let cues = []
          let track = null
          let reached = false
          try {
            const page = await watchPage(id)
            reached = true
            details = page.player ? parseDetails(page.player, page.initialData) : null
            track = pickTrack(details?.captions ?? [], args.lang)
            if (track) cues = await fetchCues(track)
          } catch (err) {
            console.warn(`[jarvis] youtube watch page for ${id}:`, err?.message ?? err)
          }
          if (!cues.length) {
            try {
              const tracks = await androidTracks(id)
              reached = true
              track = pickTrack(tracks, args.lang)
              if (track) cues = await fetchCues(track)
            } catch (err) {
              console.warn(`[jarvis] youtube android tracks for ${id}:`, err?.message ?? err)
            }
          }
          if (!cues.length) {
            // Unreachable is not the same as uncaptioned, and saying the video
            // has no captions when the network was down would be a lie.
            if (!reached) return fail('Could not reach YouTube. Say so in one sentence.')
            return fail(
              track
                ? 'The video lists captions but YouTube returned none of them. Say so in one sentence.'
                : 'That video has no captions to read. Say so, and offer what its description says instead.',
            )
          }

          const from = Math.max(0, Number(args.from) || 0)
          const to = Number(args.to) > from ? Number(args.to) : Infinity
          const lines = groupCues(cues.filter((c) => c.at >= from && c.at <= to))
          let text = ''
          let cutAt = null
          for (const line of lines) {
            if (text.length + line.length + 1 > MAX_TRANSCRIPT_CHARS) {
              cutAt = /^\[([\d:]+)\]/.exec(line)?.[1] ?? null
              break
            }
            text += `${line}\n`
          }

          const head = {
            id,
            title: details?.title,
            channel: details?.channel,
            length: details?.seconds ? secondsToClock(details.seconds) : undefined,
            captions: `${track.lang}${track.auto ? ' (automatic)' : ''}`,
            playFromUrl: `${watchUrl(id)}&t=<seconds>s — replace <seconds> to open the player at a moment`,
            chapters: details?.chapters?.length
              ? details.chapters.map((c) => `${secondsToClock(c.at)} ${c.title}`)
              : undefined,
          }
          const tail = cutAt
            ? `\n[Transcript cut at ${cutAt} for length. Call again with from=${clockToSeconds(cutAt)} to read on.]`
            : ''
          return ok(`${JSON.stringify(head, null, 1)}\n\nTRANSCRIPT\n${text}${tail}`)
        },
      ),
    ],
  })
}
