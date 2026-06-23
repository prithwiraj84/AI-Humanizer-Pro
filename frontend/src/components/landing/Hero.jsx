"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles, ChevronDown, ShieldCheck } from "lucide-react";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, y: 26 },
  show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } },
};

const TRUST = ["Bypasses AI detectors", "Keeps your meaning", "Built for academia & work"];

export default function Hero() {
  return (
    <section className="relative flex min-h-[100svh] items-center justify-center px-5 pt-24 text-center sm:px-8">
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="container-x flex flex-col items-center"
      >
        <motion.div variants={item}>
          <span className="chip animate-pulse-glow">
            <Sparkles className="h-3.5 w-3.5 text-brand-cyan" />
            Powered by advanced linguistic AI
          </span>
        </motion.div>

        <motion.h1
          variants={item}
          className="mt-6 max-w-4xl text-balance font-display text-4xl font-bold leading-[1.05] sm:text-6xl md:text-7xl"
        >
          Make AI text sound
          <br className="hidden sm:block" />{" "}
          <span className="gradient-text-animated">authentically human</span>
        </motion.h1>

        <motion.p
          variants={item}
          className="mt-6 max-w-2xl text-balance text-base leading-relaxed text-slate-400 sm:text-lg"
        >
          AI Humanizer Pro rewrites robotic, AI-generated drafts into natural
          writing that reads like you — and slips past AI detectors. For
          students, researchers, and working professionals.
        </motion.p>

        <motion.div
          variants={item}
          className="mt-9 flex flex-col items-center gap-3 sm:flex-row"
        >
          <Link href="/signup" className="btn-primary px-7 py-3.5 text-base">
            Humanize for free
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link href="/docs" className="btn-ghost px-7 py-3.5 text-base">
            Explore the API
          </Link>
        </motion.div>

        <motion.div
          variants={item}
          className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-slate-400"
        >
          {TRUST.map((t) => (
            <span key={t} className="inline-flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5 text-brand-emerald" />
              {t}
            </span>
          ))}
        </motion.div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 text-slate-500"
      >
        <ChevronDown className="h-6 w-6 animate-bounce" />
      </motion.div>
    </section>
  );
}
