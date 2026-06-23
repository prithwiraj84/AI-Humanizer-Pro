/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#06060f",
          900: "#0a0a1a",
          850: "#0d0d22",
          800: "#12122e",
          700: "#1a1a3a",
        },
        brand: {
          indigo: "#6366f1",
          violet: "#a855f7",
          cyan: "#22d3ee",
          pink: "#ec4899",
          emerald: "#10b981",
        },
      },
      fontFamily: {
        display: ["var(--font-space)", "system-ui", "sans-serif"],
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      backgroundImage: {
        "brand-gradient":
          "linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #22d3ee 100%)",
        "brand-radial":
          "radial-gradient(circle at 50% 0%, rgba(99,102,241,0.18), transparent 60%)",
      },
      boxShadow: {
        glow: "0 0 40px -8px rgba(99,102,241,0.55)",
        "glow-cyan": "0 0 40px -8px rgba(34,211,238,0.5)",
        "glow-violet": "0 0 50px -10px rgba(168,85,247,0.55)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: 0, transform: "translateY(24px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        float: {
          "0%,100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% center" },
          "100%": { backgroundPosition: "-200% center" },
        },
        "pulse-glow": {
          "0%,100%": { opacity: 0.55 },
          "50%": { opacity: 1 },
        },
      },
      animation: {
        "fade-up": "fade-up 0.7s cubic-bezier(0.16,1,0.3,1) both",
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 6s linear infinite",
        "pulse-glow": "pulse-glow 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
