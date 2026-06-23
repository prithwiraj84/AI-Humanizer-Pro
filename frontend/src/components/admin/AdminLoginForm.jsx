"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, ShieldCheck } from "lucide-react";
import AuthShell from "@/components/auth/AuthShell";
import { api } from "@/lib/api";

export default function AdminLoginForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      await api.adminLogin(email, password);
      router.push("/admin");
    } catch (e) {
      setErr(e.message || "Invalid admin credentials");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      title="Admin console"
      subtitle="Restricted access — administrators only"
    >
      <form onSubmit={onSubmit} className="space-y-4">
        <div className="flex items-center gap-2 rounded-lg border border-brand-violet/30 bg-brand-violet/10 px-4 py-3 text-sm text-violet-200">
          <ShieldCheck className="h-4 w-4" />
          This area is monitored and logged.
        </div>
        {err && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {err}
          </div>
        )}
        <label className="block">
          <span className="mb-1.5 block text-xs font-medium text-slate-400">
            Admin email
          </span>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="field"
            placeholder="admin@example.com"
          />
        </label>
        <label className="block">
          <span className="mb-1.5 block text-xs font-medium text-slate-400">
            Password
          </span>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="field"
            placeholder="••••••••"
          />
        </label>
        <button disabled={loading} className="btn-primary w-full">
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Enter console
        </button>
      </form>
    </AuthShell>
  );
}
