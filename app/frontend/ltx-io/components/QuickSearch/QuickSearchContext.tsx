import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

type QuickSearchContextValue = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
};

const QuickSearchContext = createContext<QuickSearchContextValue | null>(null);

export function QuickSearchProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((current) => !current), []);
  const value = useMemo(
    () => ({ isOpen, open, close, toggle }),
    [close, isOpen, open, toggle],
  );
  return (
    <QuickSearchContext.Provider value={value}>{children}</QuickSearchContext.Provider>
  );
}

export function useQuickSearch(): QuickSearchContextValue {
  const value = useContext(QuickSearchContext);
  if (value == null) {
    throw new Error("useQuickSearch must be used within QuickSearchProvider");
  }
  return value;
}
