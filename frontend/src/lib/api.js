// Thin client for the FastAPI backend. Every call goes through the Next.js
// rewrite at /backend/* so the session cookie stays same-origin.
const BASE = "/backend";

async function request(path, { method = "GET", body, headers = {} } = {}) {
  const opts = { method, credentials: "include", headers: { ...headers } };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }

  const res = await fetch(`${BASE}${path}`, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json")
    ? await res.json().catch(() => null)
    : null;

  if (!res.ok) {
    const message =
      (data && (data.message || data.error)) || `Request failed (${res.status})`;
    const err = new Error(message);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

export const api = {
  login: (email, password) =>
    request("/login", { method: "POST", body: { email, password } }),
  signup: (username, email, password) =>
    request("/signup", { method: "POST", body: { username, email, password } }),
  logout: () => request("/auth/logout", { method: "POST" }),
  checkAuth: () => request("/auth/check"),
  sendOtp: (email) =>
    request("/api/send-otp", { method: "POST", body: { email } }),
  resetPassword: (email, otp, new_password) =>
    request("/api/reset-password", {
      method: "POST",
      body: { email, otp, new_password },
    }),
  humanize: (payload) =>
    request("/humanize", { method: "POST", body: payload }),
  getKeys: () => request("/api/keys"),
  createKey: (name) =>
    request("/api/keys", { method: "POST", body: { name } }),
  deleteKey: (id) => request(`/api/keys/${id}`, { method: "DELETE" }),
  getUsage: () => request("/api/usage"),

  // ---- Admin ----
  adminLogin: (email, password) =>
    request("/admin/login", { method: "POST", body: { email, password } }),
  adminLogout: () => request("/admin/logout", { method: "POST" }),
  adminCheck: () => request("/admin/auth/check"),
  adminOverview: () => request("/admin/api/overview"),
  adminUsers: () => request("/admin/api/users"),
  adminUserUsage: (id) => request(`/admin/api/users/${id}/usage`),
  adminDeleteUser: (id) =>
    request(`/admin/api/users/${id}`, { method: "DELETE" }),
  adminKeys: () => request("/admin/api/keys"),
  adminDeleteKey: (id) =>
    request(`/admin/api/keys/${id}`, { method: "DELETE" }),
  adminActivityLog: (limit = 200, includeAdmin = false) =>
    request(
      `/admin/api/activity-log?limit=${limit}&include_admin=${includeAdmin}`
    ),
};
