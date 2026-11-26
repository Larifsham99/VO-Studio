// Talks to the home server that serves this page. The token arrives once via
// the link the server prints (#token=...) and is then kept on this phone.

const KEY = 'latif_token'

function readHashToken() {
  const m = window.location.hash.match(/token=([^&]+)/)
  if (m) {
    try { localStorage.setItem(KEY, decodeURIComponent(m[1])) } catch {}
    history.replaceState(null, '', window.location.pathname)
  }
}
readHashToken()

export function getToken() {
  try { return localStorage.getItem(KEY) || '' } catch { return '' }
}
export function setToken(t) {
  try { t ? localStorage.setItem(KEY, t) : localStorage.removeItem(KEY) } catch {}
}

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status }
}

async function req(method, path, body) {
  const opts = { method, headers: { Authorization: `Bearer ${getToken()}` } }
  if (body instanceof FormData) opts.body = body
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  let r
  try { r = await fetch(path, opts) } catch {
    throw new ApiError(0, 'Tidak dapat menghubungi PC. Pastikan server hidup dan fon bersambung (Wi-Fi rumah atau Tailscale).')
  }
  const data = await r.json().catch(() => ({}))
  if (!r.ok) {
    let msg = data.detail
    if (Array.isArray(msg)) msg = msg.map(d => d.msg).join('; ')
    throw new ApiError(r.status, msg || `Ralat ${r.status}`)
  }
  return data
}

export const api = {
  status: () => req('GET', '/api/status'),
  loadModel: name => req('POST', `/api/model/${name}`),
  projects: () => req('GET', '/api/projects'),
  createProject: p => req('POST', '/api/projects', p),
  deleteProject: id => req('DELETE', `/api/projects/${id}`),
  references: () => req('GET', '/api/references'),
  uploadReference: form => req('POST', '/api/references', form),
  editReference: (id, body) => req('PATCH', `/api/references/${id}`, body),
  deleteReference: id => req('DELETE', `/api/references/${id}`),
  generate: g => req('POST', '/api/generate', g),
  jobs: projectId => req('GET', `/api/jobs?project_id=${projectId}`),
  deleteJob: id => req('DELETE', `/api/jobs/${id}`),
}

const tq = () => `token=${encodeURIComponent(getToken())}`
export const audioUrl = (id, fmt = 'wav', download = false) =>
  `/api/jobs/${id}/audio?format=${fmt}${download ? '&download=true' : ''}&${tq()}`
export const refAudioUrl = id => `/api/references/${id}/audio?${tq()}`
