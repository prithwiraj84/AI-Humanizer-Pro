import { ArrowRight, Bot, User } from "lucide-react";
import Link from "next/link";
import Reveal from "@/components/ui/Reveal";

export default function Showcase() {
  return (
    <section className="section-pad relative">
      <div className="container-x">
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="chip">Before & after</span>
          <h2 className="mt-4 font-display text-3xl font-bold sm:text-5xl">
            See the <span className="gradient-text">transformation</span>
          </h2>
        </Reveal>

        <div className="mt-14 grid items-stretch gap-5 lg:grid-cols-[1fr_auto_1fr]">
          <Reveal>
            <div className="glass h-full rounded-2xl p-6">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-rose-500/15 text-rose-300">
                  <Bot className="h-4 w-4" />
                </span>
                AI-generated draft
              </div>
              <p className="mt-4 text-sm leading-relaxed text-slate-400">
                “The utilization of artificial intelligence facilitates the
                optimization of content generation processes, thereby enabling
                users to achieve enhanced productivity outcomes in a
                time-efficient manner.”
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-lg bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-300">
                AI probability · 98%
              </div>
            </div>
          </Reveal>

          <div className="hidden items-center justify-center lg:flex">
            <span className="grid h-12 w-12 place-items-center rounded-full bg-brand-gradient shadow-glow">
              <ArrowRight className="h-5 w-5 text-white" />
            </span>
          </div>

          <Reveal delay={0.12}>
            <div className="glass h-full rounded-2xl p-6 ring-1 ring-brand-emerald/20">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-500/15 text-emerald-300">
                  <User className="h-4 w-4" />
                </span>
                Humanized output
              </div>
              <p className="mt-4 text-sm leading-relaxed text-slate-200">
                “Using AI to help write content just makes the whole thing
                faster. You get more done without burning hours on it — and it
                still sounds like something you'd actually say.”
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-300">
                AI probability · 3%
              </div>
            </div>
          </Reveal>
        </div>

        <Reveal className="mt-10 text-center">
          <Link href="/humanizer" className="btn-primary px-7 py-3.5 text-base">
            Try it on your text
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Reveal>
      </div>
    </section>
  );
}
