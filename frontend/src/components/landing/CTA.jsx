import Link from "next/link";
import { ArrowRight } from "lucide-react";
import Reveal from "@/components/ui/Reveal";

export default function CTA() {
  return (
    <section className="section-pad relative">
      <div className="container-x">
        <Reveal>
          <div className="relative overflow-hidden rounded-3xl border border-white/10 p-10 text-center sm:p-16">
            <div className="absolute inset-0 -z-10 bg-brand-gradient opacity-90" />
            <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.25),transparent_45%)]" />
            <h2 className="mx-auto max-w-2xl text-balance font-display text-3xl font-bold text-white sm:text-5xl">
              Ready to make your writing sound human?
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-balance text-white/85">
              Create a free account and humanize your first draft in under a
              minute. No credit card required.
            </p>
            <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link
                href="/signup"
                className="btn bg-white px-7 py-3.5 text-base font-semibold text-ink-900 hover:-translate-y-0.5 hover:bg-slate-100"
              >
                Get started free
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/docs"
                className="btn border border-white/40 px-7 py-3.5 text-base font-semibold text-white hover:bg-white/10"
              >
                Read the docs
              </Link>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
