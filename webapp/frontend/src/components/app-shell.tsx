import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import { useViewStore, type AppView } from "@/store/view-store";

const TABS: Array<{ id: AppView; label: string }> = [
  { id: "design", label: "Current Design" },
  { id: "sweep", label: "Parametric Sweep" },
  { id: "pareto", label: "Optimize Design" },
  { id: "shap", label: "Explain Design" },
];

/** Top-level layout: header with status badge + body slot. */
export function AppShell({ children }: { children: ReactNode }) {
  const view = useViewStore((s) => s.view);
  const setView = useViewStore((s) => s.setView);

  return (
    <div className="min-h-screen">
      <header className="border-b bg-[var(--color-card)]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              RoverDevKit
            </h1>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              Interactive design-space explorer for lunar micro-rovers.
            </p>
          </div>
        </div>
        <nav className="mx-auto flex max-w-6xl gap-1 px-6 pb-2 text-sm">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setView(tab.id)}
              aria-pressed={view === tab.id}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm transition-colors",
                view === tab.id
                  ? "bg-[var(--color-accent)] text-[var(--color-accent-foreground)]"
                  : "text-[var(--color-muted-foreground)] hover:bg-[var(--color-muted)]",
              )}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
      <footer className="mx-auto max-w-6xl px-6 py-6 text-xs text-[var(--color-muted-foreground)]">
        Collaborative Autonomous Systems Lab · Duke University
      </footer>
    </div>
  );
}
