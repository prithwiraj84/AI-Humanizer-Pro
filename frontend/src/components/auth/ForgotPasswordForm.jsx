"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, CheckCircle2 } from "lucide-react";
import AuthShell from "./AuthShell";
import { api } from "@/lib/api";

export default function ForgotPasswordForm() {
  const router = useRouter();
  const [step, setStep] = useState(1); // 1 = request OTP, 2 = reset
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  async function requestOtp(e) {
    e.preventDefault();
    setErr("");
    setMsg("");
    setLoading(true);
    try {
      const res = await api.sendOtp(email);
      setMsg(res?.message || "If this email exists, a code has been sent.");
      setStep(2);
    } catch (e) {
      setErr(e.message || "Could not send code");
    } finally {
      setLoading(false);
    }
  }

  async function resetPassword(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      await api.resetPassword(email, otp, newPassword);
      setDone(true);
      setTimeout(() => router.push("/login"), 1500);
    } catch (e) {
      setErr(e.message || "Could not reset password");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      title="Reset password"
      subtitle={
        step === 1
          ? "Enter your email to receive a one-time code"
          : "Enter the code and your new password"
      }
      footer={
        <>
          Remembered it?{" "}
          <Link href="/login" className="font-medium text-indigo-300 hover:underline">
            Back to log in
          </Link>
        </>
      }
    >
      {done ? (
        <div className="flex flex-col items-center gap-3 py-6 text-center">
          <CheckCircle2 className="h-12 w-12 text-brand-emerald" />
          <p className="font-display text-lg font-semibold">Password updated!</p>
          <p className="text-sm text-slate-400">Redirecting to log in…</p>
        </div>
      ) : step === 1 ? (
        <form onSubmit={requestOtp} className="space-y-4">
          {err && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              {err}
            </div>
          )}
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
          <button disabled={loading} className="btn-primary w-full">
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Send code
          </button>
        </form>
      ) : (
        <form onSubmit={resetPassword} className="space-y-4">
          {msg && (
            <div className="rounded-lg border border-brand-indigo/30 bg-brand-indigo/10 px-4 py-3 text-sm text-indigo-200">
              {msg}
            </div>
          )}
          {err && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              {err}
            </div>
          )}
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">
              6-digit code
            </span>
            <input
              required
              inputMode="numeric"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="field tracking-[0.4em]"
              placeholder="••••••"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">
              New password
            </span>
            <input
              type="password"
              required
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="field"
              placeholder="New strong password"
            />
          </label>
          <button disabled={loading} className="btn-primary w-full">
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Reset password
          </button>
        </form>
      )}
    </AuthShell>
  );
}
