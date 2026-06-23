import Protected from "@/components/auth/Protected";
import DashboardView from "@/components/dashboard/DashboardView";

export const metadata = {
  title: "Dashboard",
  robots: { index: false, follow: false },
};

export default function DashboardPage() {
  return (
    <Protected>
      <DashboardView />
    </Protected>
  );
}
