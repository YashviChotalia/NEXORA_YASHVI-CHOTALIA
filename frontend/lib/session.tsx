"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { authApi, type Me } from "@/lib/api";

interface Ctx { me: Me | null; ready: boolean; refresh: () => Promise<void>; can: (perm: string) => boolean }
const SessionCtx = createContext<Ctx>({ me: null, ready: false, refresh: async () => {}, can: () => false });
export const PUBLIC_PATHS = ["/login", "/setup"];
/** Open to everyone and never redirected (the public landing page). */
export const OPEN_PATHS = ["/welcome"];

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [ready, setReady] = useState(false);
  const router = useRouter();
  const path = usePathname();

  const refresh = useCallback(async () => {
    if (OPEN_PATHS.some((p) => path.startsWith(p))) { setReady(true); return; }
    try {
      const s = await authApi.state();
      setMe(s.user || null);
      const isPublic = PUBLIC_PATHS.some((p) => path.startsWith(p));
      if (s.setup_required) { if (!path.startsWith("/setup")) router.replace("/setup"); }
      // signed out: the home address opens the landing page; a deep link goes to sign-in and comes back after
      else if (!s.signed_in) { if (!isPublic) router.replace(path === "/" ? "/welcome" : `/login?next=${encodeURIComponent(path)}`); }
      else if (s.user?.must_change_password && !path.startsWith("/account")) router.replace("/account?required=1");
      else if (isPublic) router.replace(s.user?.permissions.includes("case:read") ? "/" : "/admin/users");
    } catch {
      setMe(null);
    } finally {
      setReady(true);
    }
  }, [path, router]);

  useEffect(() => { refresh(); }, [refresh]);

  const can = (perm: string) => !!me?.permissions.includes(perm);
  return <SessionCtx.Provider value={{ me, ready, refresh, can }}>{children}</SessionCtx.Provider>;
}

export const useSession = () => useContext(SessionCtx);
