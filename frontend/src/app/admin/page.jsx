import AdminProtected from "@/components/admin/AdminProtected";
import AdminDashboard from "@/components/admin/AdminDashboard";

export const metadata = {
  title: "Admin Console",
  robots: { index: false, follow: false },
};

export default function AdminPage() {
  return (
    <AdminProtected>
      <AdminDashboard />
    </AdminProtected>
  );
}
