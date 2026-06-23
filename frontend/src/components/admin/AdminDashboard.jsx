"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  LayoutDashboard,
  Users,
  KeyRound,
  Activity,
  LogOut,
  Trash2,
  Eye,
  X,
  ShieldCheck,
  Loader2,
  Sparkles,
  RefreshCw,
} from "lucide-react";
import { api } from "@/lib/api";

const TABS = [
  { key: "overview", label: "Overview", icon: LayoutDashboard },
  { key: "users", label: "Users", icon: Users },
  { key: "keys", label: "API Keys", icon: KeyRound },
  { key: "activity", label: "Activity Log", icon: Activity },
];

function StatCard({ label, value }) {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="mt-2 font-display text-2xl font-bold gradient-text">
        {value}
      </div>
    </div>
  );
}

function Th({ children }) {
  return <th className="pb-3 pr-4 font-medium">{children}</th>;
}

export default function AdminDashboard() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [overview, setOverview] = useState(null);
  const [users, setUsers] = useState(null);
  const [keys, setKeys] = useState(null);
  const [activity, setActivity] = useState(null);
  const [includeAdmin, setIncludeAdmin] = useState(false);
  const [usage, setUsage] = useState(null); // user-usage modal

  const guard = useCallback(
    (e) => {
      if (e?.status === 401) {
        router.replace("/admin/login");
        return true;
      }
      setErr(e?.message || "Something went wrong");
      return false;
    },
    [router]
  );

  const loadOverview = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      setOverview(await api.adminOverview());
    } catch (e) {
      guard(e);
    } finally {
      setLoading(false);
    }
  }, [guard]);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      setUsers((await api.adminUsers()).users || []);
    } catch (e) {
      guard(e);
    } finally {
      setLoading(false);
    }
  }, [guard]);

  const loadKeys = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      setKeys((await api.adminKeys()).keys || []);
    } catch (e) {
      guard(e);
    } finally {
      setLoading(false);
    }
  }, [guard]);

  const loadActivity = useCallback(
    async (incAdmin) => {
      setLoading(true);
      setErr("");
      try {
        setActivity((await api.adminActivityLog(200, incAdmin)).logs || []);
      } catch (e) {
        guard(e);
      } finally {
        setLoading(false);
      }
    },
    [guard]
  );

  // Initial: grab admin email + overview.
  useEffect(() => {
    api
      .adminCheck()
      .then((d) => setEmail(d?.email || ""))
      .catch(() => router.replace("/admin/login"));
    loadOverview();
  }, [loadOverview, router]);

  // Lazy-load a section the first time its tab is opened.
  useEffect(() => {
    if (tab === "users" && users === null) loadUsers();
    if (tab === "keys" && keys === null) loadKeys();
    if (tab === "activity" && activity === null) loadActivity(includeAdmin);
  }, [tab, users, keys, activity, includeAdmin, loadUsers, loadKeys, loadActivity]);

  async function deleteUser(u) {
    if (!confirm(`Delete user "${u.username}" and all their API data?`)) return;
    setUsers((cur) => cur.filter((x) => x.id !== u.id));
    try {
      await api.adminDeleteUser(u.id);
    } catch (e) {
      if (!guard(e)) loadUsers();
    }
  }

  async function revokeKey(k) {
    if (!confirm(`Delete API key "${k.key_name}"?`)) return;
    setKeys((cur) => cur.filter((x) => x.id !== k.id));
    try {
      await api.adminDeleteKey(k.id);
    } catch (e) {
      if (!guard(e)) loadKeys();
    }
  }

  async function openUsage(u) {
    try {
      setUsage({ loading: true, user: u });
      const data = await api.adminUserUsage(u.id);
      setUsage({ loading: false, ...data });
    } catch (e) {
      setUsage(null);
      guard(e);
    }
  }

  async function logout() {
    try {
      await api.adminLogout();
    } catch {
      /* ignore */
    }
    router.replace("/admin/login");
  }

  return (
    <div className="min-h-screen">
      {/* Top bar */}
      <header className="sticky top-0 z-40 border-b border-white/10 bg-ink-950/70 backdrop-blur-xl">
        <div className="container-x flex h-16 items-center justify-between px-5 sm:px-8">
          <Link href="/" className="flex items-center gap-2 font-display font-bold">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-gradient shadow-glow">
              <Sparkles className="h-5 w-5 text-white" />
            </span>
            Admin <span className="gradient-text">Console</span>
          </Link>
          <div className="flex items-center gap-3">
            <span className="hidden items-center gap-1.5 text-xs text-slate-400 sm:flex">
              <ShieldCheck className="h-3.5 w-3.5 text-brand-emerald" />
              {email}
            </span>
            <button onClick={logout} className="btn-ghost">
              <LogOut className="h-4 w-4" />
              Log out
            </button>
          </div>
        </div>
      </header>

      <main className="container-x px-5 pb-24 pt-8 sm:px-8">
        {/* Tabs */}
        <div className="mb-6 flex flex-wrap gap-2">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`btn gap-2 ${
                tab === t.key
                  ? "bg-brand-gradient text-white shadow-glow"
                  : "border border-white/10 bg-white/5 text-slate-300 hover:text-white"
              }`}
            >
              <t.icon className="h-4 w-4" />
              {t.label}
            </button>
          ))}
        </div>

        {err && (
          <div className="mb-5 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {err}
          </div>
        )}

        {loading && (
          <div className="grid min-h-[30vh] place-items-center">
            <Loader2 className="h-8 w-8 animate-spin text-brand-indigo" />
          </div>
        )}

        {/* OVERVIEW */}
        {!loading && tab === "overview" && overview && (
          <div className="space-y-8">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard label="Total users" value={overview.stats.total_users} />
              <StatCard label="API keys" value={overview.stats.total_api_keys} />
              <StatCard label="Active keys" value={overview.stats.active_api_keys} />
              <StatCard label="Total requests" value={overview.stats.total_requests} />
              <StatCard label="Requests (24h)" value={overview.stats.requests_24h} />
              <StatCard label="Avg latency" value={`${overview.stats.avg_latency_ms} ms`} />
              <StatCard label="Success rate" value={`${overview.stats.success_rate}%`} />
            </div>

            <section className="glass-strong rounded-2xl p-6">
              <h2 className="font-display text-lg font-semibold">Top endpoints</h2>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-wider text-slate-400">
                    <tr><Th>Endpoint</Th><Th>Requests</Th><Th>Success</Th><Th>Avg latency</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {overview.top_endpoints.map((r, i) => (
                      <tr key={i} className="text-slate-300">
                        <td className="py-3 pr-4 font-mono text-xs">{r.endpoint}</td>
                        <td className="py-3 pr-4">{r.total_requests}</td>
                        <td className="py-3 pr-4">{r.success_rate}%</td>
                        <td className="py-3">{r.avg_latency_ms} ms</td>
                      </tr>
                    ))}
                    {overview.top_endpoints.length === 0 && (
                      <tr><td className="py-3 text-slate-500" colSpan={4}>No data yet.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="glass-strong rounded-2xl p-6">
              <h2 className="font-display text-lg font-semibold">Recent activity</h2>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-wider text-slate-400">
                    <tr><Th>Time</Th><Th>User</Th><Th>Endpoint</Th><Th>Status</Th><Th>Latency</Th></tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {overview.recent_activity.map((r, i) => (
                      <tr key={i} className="text-slate-300">
                        <td className="py-3 pr-4 text-slate-400">{r.timestamp}</td>
                        <td className="py-3 pr-4">{r.username || "—"}</td>
                        <td className="py-3 pr-4 font-mono text-xs">{r.endpoint}</td>
                        <td className="py-3 pr-4">
                          <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${r.status === "success" ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/15 text-rose-300"}`}>{r.status}</span>
                        </td>
                        <td className="py-3">{r.response_time} ms</td>
                      </tr>
                    ))}
                    {overview.recent_activity.length === 0 && (
                      <tr><td className="py-3 text-slate-500" colSpan={5}>No activity yet.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* USERS */}
        {!loading && tab === "users" && users && (
          <section className="glass-strong rounded-2xl p-6">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-lg font-semibold">Users ({users.length})</h2>
              <button onClick={loadUsers} className="btn-ghost"><RefreshCw className="h-4 w-4" />Refresh</button>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-400">
                  <tr><Th>User</Th><Th>Email</Th><Th>Keys</Th><Th>Requests</Th><Th>Success</Th><Th>Joined</Th><Th>Actions</Th></tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {users.map((u) => (
                    <tr key={u.id} className="text-slate-300">
                      <td className="py-3 pr-4 font-medium text-white">{u.username}</td>
                      <td className="py-3 pr-4">{u.email}</td>
                      <td className="py-3 pr-4">{u.api_key_count}</td>
                      <td className="py-3 pr-4">{u.total_requests}</td>
                      <td className="py-3 pr-4">{u.success_rate}%</td>
                      <td className="py-3 pr-4 text-slate-400">{u.created_at}</td>
                      <td className="py-3">
                        <div className="flex gap-2">
                          <button onClick={() => openUsage(u)} aria-label={`View usage for ${u.username}`} className="btn-ghost px-3 py-2"><Eye className="h-4 w-4" /></button>
                          <button onClick={() => deleteUser(u)} aria-label={`Delete user ${u.username}`} className="btn px-3 py-2 text-rose-300 hover:bg-rose-500/10"><Trash2 className="h-4 w-4" /></button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {users.length === 0 && (
                    <tr><td className="py-3 text-slate-500" colSpan={7}>No users.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* KEYS */}
        {!loading && tab === "keys" && keys && (
          <section className="glass-strong rounded-2xl p-6">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-lg font-semibold">API Keys ({keys.length})</h2>
              <button onClick={loadKeys} className="btn-ghost"><RefreshCw className="h-4 w-4" />Refresh</button>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-400">
                  <tr><Th>Name</Th><Th>Key</Th><Th>Owner</Th><Th>Requests</Th><Th>Last used</Th><Th>Actions</Th></tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {keys.map((k) => (
                    <tr key={k.id} className="text-slate-300">
                      <td className="py-3 pr-4 font-medium text-white">{k.key_name}</td>
                      <td className="py-3 pr-4 font-mono text-xs text-slate-400">{k.api_key.slice(0, 16)}…</td>
                      <td className="py-3 pr-4">{k.username}</td>
                      <td className="py-3 pr-4">{k.request_count}</td>
                      <td className="py-3 pr-4 text-slate-400">{k.last_used || "—"}</td>
                      <td className="py-3">
                        <button onClick={() => revokeKey(k)} aria-label={`Delete key ${k.key_name}`} className="btn px-3 py-2 text-rose-300 hover:bg-rose-500/10"><Trash2 className="h-4 w-4" /></button>
                      </td>
                    </tr>
                  ))}
                  {keys.length === 0 && (
                    <tr><td className="py-3 text-slate-500" colSpan={6}>No API keys.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ACTIVITY */}
        {!loading && tab === "activity" && activity && (
          <section className="glass-strong rounded-2xl p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="font-display text-lg font-semibold">Activity Log ({activity.length})</h2>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={includeAdmin}
                  onChange={(e) => {
                    setIncludeAdmin(e.target.checked);
                    setActivity(null);
                    loadActivity(e.target.checked);
                  }}
                  className="h-4 w-4 rounded border-white/20 bg-ink-900"
                />
                Include admin events
              </label>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-400">
                  <tr><Th>Time</Th><Th>Actor</Th><Th>Action</Th><Th>Endpoint</Th><Th>Method</Th><Th>Status</Th><Th>IP</Th></tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {activity.map((l) => (
                    <tr key={l.id} className="text-slate-300">
                      <td className="py-3 pr-4 text-slate-400">{l.created_at}</td>
                      <td className="py-3 pr-4">
                        <span className="font-medium text-white">{l.username || l.actor_identifier || "—"}</span>
                        <span className="ml-1 text-xs text-slate-500">({l.actor_type})</span>
                      </td>
                      <td className="py-3 pr-4 font-mono text-xs">{l.action}</td>
                      <td className="py-3 pr-4 font-mono text-xs">{l.endpoint}</td>
                      <td className="py-3 pr-4">{l.method}</td>
                      <td className="py-3 pr-4">
                        <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${l.status_code < 400 ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/15 text-rose-300"}`}>{l.status_code}</span>
                      </td>
                      <td className="py-3 text-slate-400">{l.ip_address || "—"}</td>
                    </tr>
                  ))}
                  {activity.length === 0 && (
                    <tr><td className="py-3 text-slate-500" colSpan={7}>No log entries.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      {/* User usage modal */}
      {usage && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4 backdrop-blur-sm"
          onClick={() => setUsage(null)}
        >
          <div
            className="glass-strong max-h-[85vh] w-full max-w-2xl overflow-auto rounded-2xl p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-display text-lg font-semibold">
                Usage · {usage.user?.username}
              </h3>
              <button onClick={() => setUsage(null)} aria-label="Close" className="btn-ghost px-3 py-2"><X className="h-4 w-4" /></button>
            </div>
            {usage.loading ? (
              <div className="grid h-40 place-items-center"><Loader2 className="h-7 w-7 animate-spin text-brand-indigo" /></div>
            ) : (
              <div className="mt-4 space-y-6 text-sm">
                <div className="text-slate-400">{usage.user?.email}</div>
                <div>
                  <h4 className="mb-2 font-semibold text-white">API keys</h4>
                  {usage.keys?.length ? usage.keys.map((k) => (
                    <div key={k.id} className="flex justify-between border-b border-white/5 py-2">
                      <span>{k.key_name}</span>
                      <span className="text-slate-400">{k.request_count} req · {k.success_rate}%</span>
                    </div>
                  )) : <p className="text-slate-500">No keys.</p>}
                </div>
                <div>
                  <h4 className="mb-2 font-semibold text-white">Endpoints</h4>
                  {usage.endpoint_breakdown?.length ? usage.endpoint_breakdown.map((e, i) => (
                    <div key={i} className="flex justify-between border-b border-white/5 py-2">
                      <span className="font-mono text-xs">{e.endpoint}</span>
                      <span className="text-slate-400">{e.total_requests} · {e.avg_latency_ms} ms</span>
                    </div>
                  )) : <p className="text-slate-500">No requests.</p>}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
