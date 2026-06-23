"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  Gauge,
  CheckCircle2,
  KeyRound,
  Plus,
  Trash2,
  Copy,
  Check,
  Loader2,
  Wand2,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

function StatCard({ icon: Icon, label, value, tone }) {
  const tones = {
    indigo: "from-brand-indigo to-brand-violet",
    cyan: "from-brand-cyan to-brand-indigo",
    emerald: "from-brand-emerald to-brand-cyan",
  };
  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-400">{label}</span>
        <span className={`grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br ${tones[tone]}`}>
          <Icon className="h-4 w-4 text-white" />
        </span>
      </div>
      <div className="mt-3 font-display text-3xl font-bold">{value}</div>
    </div>
  );
}

export default function DashboardView() {
  const { user } = useAuth();
  const router = useRouter();
  const [usage, setUsage] = useState(null);
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      const [u, k] = await Promise.all([api.getUsage(), api.getKeys()]);
      setUsage(u);
      setKeys(k?.keys || []);
    } catch (e) {
      if (e.status === 401) {
        router.replace("/login");
        return;
      }
      setErr(e.message || "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    load();
  }, [load]);

  async function createKey(e) {
    e.preventDefault();
    setCreating(true);
    setErr("");
    try {
      await api.createKey(newName || "My API Key");
      setNewName("");
      await load();
    } catch (e) {
      if (e.status === 401) {
        router.replace("/login");
        return;
      }
      setErr(e.message || "Could not create key");
    } finally {
      setCreating(false);
    }
  }

  async function removeKey(id) {
    setKeys((ks) => ks.filter((k) => k.id !== id));
    try {
      await api.deleteKey(id);
    } catch {
      load();
    }
  }

  async function copyKey(k) {
    await navigator.clipboard.writeText(k.api_key);
    setCopiedId(k.id);
    setTimeout(() => setCopiedId(null), 1600);
  }

  return (
    <>
      <Navbar />
      <main className="container-x px-5 pb-24 pt-28 sm:px-8">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-3xl font-bold sm:text-4xl">
              Welcome back{user?.username ? `, ${user.username}` : ""}
            </h1>
            <p className="mt-2 text-slate-400">
              Monitor your usage and manage your API keys.
            </p>
          </div>
          <Link href="/humanizer" className="btn-primary">
            <Wand2 className="h-4 w-4" />
            Open humanizer
          </Link>
        </div>

        {loading ? (
          <div className="grid min-h-[40vh] place-items-center">
            <Loader2 className="h-8 w-8 animate-spin text-brand-indigo" />
          </div>
        ) : (
          <div className="space-y-8">
            {err && (
              <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
                {err}
              </div>
            )}

            {/* Stats */}
            <div className="grid gap-5 sm:grid-cols-3">
              <StatCard
                icon={Activity}
                label="Total requests"
                value={usage?.stats?.total_requests ?? 0}
                tone="indigo"
              />
              <StatCard
                icon={CheckCircle2}
                label="Success rate"
                value={usage?.stats?.success_rate ?? "100%"}
                tone="emerald"
              />
              <StatCard
                icon={Gauge}
                label="Avg. latency"
                value={`${usage?.stats?.avg_latency_ms ?? 0} ms`}
                tone="cyan"
              />
            </div>

            {/* API Keys */}
            <section className="glass-strong rounded-2xl p-6">
              <div className="flex items-center gap-2">
                <KeyRound className="h-5 w-5 text-brand-violet" />
                <h2 className="font-display text-xl font-semibold">API Keys</h2>
              </div>

              <form onSubmit={createKey} className="mt-5 flex flex-wrap gap-3">
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Key name (e.g. Production)"
                  aria-label="API key name"
                  className="field max-w-xs flex-1"
                />
                <button disabled={creating} className="btn-primary">
                  {creating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                  Create key
                </button>
              </form>

              <div className="mt-5 space-y-3">
                {keys.length === 0 ? (
                  <p className="text-sm text-slate-400">
                    No API keys yet. Create one to use the API.
                  </p>
                ) : (
                  keys.map((k) => (
                    <div
                      key={k.id}
                      className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-ink-900/50 px-4 py-3"
                    >
                      <div className="min-w-0">
                        <div className="text-sm font-semibold">{k.name}</div>
                        <code className="block truncate font-mono text-xs text-slate-400">
                          {k.api_key}
                        </code>
                      </div>
                      <div className="flex shrink-0 gap-2">
                        <button
                          onClick={() => copyKey(k)}
                          aria-label={`Copy API key ${k.name}`}
                          className="btn-ghost px-3 py-2"
                        >
                          {copiedId === k.id ? (
                            <Check className="h-4 w-4 text-brand-emerald" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </button>
                        <button
                          onClick={() => removeKey(k.id)}
                          aria-label={`Delete API key ${k.name}`}
                          className="btn px-3 py-2 text-rose-300 hover:bg-rose-500/10"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>

            {/* Recent activity */}
            <section className="glass-strong rounded-2xl p-6">
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-brand-cyan" />
                <h2 className="font-display text-xl font-semibold">
                  Recent activity
                </h2>
              </div>

              <div className="mt-5 overflow-x-auto">
                {usage?.logs?.length ? (
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase tracking-wider text-slate-400">
                      <tr>
                        <th className="pb-3 pr-4 font-medium">Time</th>
                        <th className="pb-3 pr-4 font-medium">Endpoint</th>
                        <th className="pb-3 pr-4 font-medium">Status</th>
                        <th className="pb-3 font-medium">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {usage.logs.map((log, i) => (
                        <tr
                          key={`${log.timestamp}-${log.endpoint}-${i}`}
                          className="text-slate-300"
                        >
                          <td className="py-3 pr-4 text-slate-400">{log.timestamp}</td>
                          <td className="py-3 pr-4 font-mono text-xs">{log.endpoint}</td>
                          <td className="py-3 pr-4">
                            <span
                              className={`rounded-md px-2 py-0.5 text-xs font-semibold ${
                                log.status_code === 200
                                  ? "bg-emerald-500/15 text-emerald-300"
                                  : "bg-rose-500/15 text-rose-300"
                              }`}
                            >
                              {log.status_code}
                            </span>
                          </td>
                          <td className="py-3">{log.latency_ms} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-sm text-slate-400">
                    No requests yet. Humanize some text or call the API to see
                    activity here.
                  </p>
                )}
              </div>
            </section>
          </div>
        )}
      </main>
    </>
  );
}
