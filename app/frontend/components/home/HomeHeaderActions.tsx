import type { ReactNode } from "react";

import { Button } from "@/ds/Button/Button";

import { HOME_EXTERNAL_URLS, openHomeExternalUrl } from "./home-external-links";

type HomeHeaderActionsProps = {
  queueControl: ReactNode;
};

export function HomeHeaderActions({ queueControl }: HomeHeaderActionsProps) {
  return (
    <>
      {queueControl}
      <Button
        label="Contact Sales"
        appearance="neutral"
        hierarchy="secondary"
        onClick={() => openHomeExternalUrl(HOME_EXTERNAL_URLS.contactSales)}
      />
    </>
  );
}
