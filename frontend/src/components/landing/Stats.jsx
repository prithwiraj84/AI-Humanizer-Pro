import Reveal from "@/components/ui/Reveal";
import CountUp from "@/components/ui/CountUp";

const STATS = [
  { to: 2, suffix: "M+", label: "Words humanized" },
  { to: 98, suffix: "%", label: "Avg. detector drop" },
  { to: 50, suffix: "ms", label: "Median latency" },
  { to: 4.9, suffix: "/5", decimals: 1, label: "User rating" },
];

export default function Stats() {
  return (
    <section id="stats" className="relative px-5 py-16 sm:px-8">
      <div className="container-x">
        <Reveal>
          <div className="glass-strong grid gap-8 rounded-3xl p-8 sm:grid-cols-2 lg:grid-cols-4 lg:p-12">
            {STATS.map((s) => (
              <div key={s.label} className="text-center">
                <div className="font-display text-4xl font-bold gradient-text sm:text-5xl">
                  <CountUp to={s.to} suffix={s.suffix} decimals={s.decimals || 0} />
                </div>
                <div className="mt-2 text-sm text-slate-400">{s.label}</div>
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
