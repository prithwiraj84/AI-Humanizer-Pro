"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, CheckCircle2 } from "lucide-react";
import AuthShell from "./AuthShell";
import { useAuth } from "@/context/AuthContext";

export default function SignupForm() {
  const { signup } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      await signup(username, email, password);
      setDone(true);
      setTimeout(() => router.push("/login"), 1400);
    } catch (e) {
      setErr(e.message || "Could not create account");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      title="Create your account"
      subtitle="Start humanizing in under a minute"
      footer={
        <>
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-indigo-300 hover:underline">
            Log in
          </Link>
        </>
      }
    >
      {done ? (
        <div className="flex flex-col items-center gap-3 py-6 text-center">
          <CheckCircle2 className="h-12 w-12 text-brand-emerald" />
          <p className="font-display text-lg font-semibold">Account created!</p>
          <p className="text-sm text-slate-400">Redirecting you to log in…</p>
        </div>
      ) : (
        <form onSubmit={onSubmit} className="space-y-4">
          {err && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              {err}
            </div>
          )}
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">Username</span>
            <input
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="field"
              placeholder="jane_doe"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="field"
              placeholder="you@example.com"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">Password</span>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="field"
              placeholder="Create a strong password"
            />
          </label>
          <button disabled={loading} className="btn-primary w-full">
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Create account
          </button>
        </form>
      )}
    </AuthShell>
  );
}
