import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import SceneBackground from "@/components/three/SceneBackground";
import Hero from "@/components/landing/Hero";
import Features from "@/components/landing/Features";
import HowItWorks from "@/components/landing/HowItWorks";
import Showcase from "@/components/landing/Showcase";
import Audience from "@/components/landing/Audience";
import Stats from "@/components/landing/Stats";
import CTA from "@/components/landing/CTA";

export default function HomePage() {
  return (
    <>
      <SceneBackground />
      <Navbar />
      <main className="relative z-10">
        <Hero />
        <Features />
        <HowItWorks />
        <Showcase />
        <Audience />
        <Stats />
        <CTA />
      </main>
      <Footer />
    </>
  );
}
