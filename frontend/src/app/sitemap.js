import { SITE } from "@/lib/site";

export default function sitemap() {
  // Only public, indexable pages belong in the sitemap (private pages are noindex).
  const routes = ["", "/docs"];
  return routes.map((route) => ({
    url: `${SITE.url}${route}`,
    lastModified: new Date(),
    changeFrequency: route === "" ? "weekly" : "monthly",
    priority: route === "" ? 1 : 0.7,
  }));
}
