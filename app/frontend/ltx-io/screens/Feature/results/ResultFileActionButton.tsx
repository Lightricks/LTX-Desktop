import { Button } from "@ds/Button/Button";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@ds/DropdownMenu/DropdownMenu";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import { useThemeRootElement } from "@ds/styles/themes/useTheme";
import { Download, FolderOpen } from "lucide-react";

import { revealInFolderLabel } from "../../../../lib/revealInFolderLabel";

import type { ResultFileAction } from "./resultFileAction";

/**
 * Reveals or downloads the file of a result. A result with two files opens a
 * menu to pick one, like the Download menu in the web app.
 */
export function ResultFileActionButton({ action }: { action: ResultFileAction }) {
  const rootElement = useThemeRootElement();
  const label = action.kind === "reveal" ? revealInFolderLabel() : "Download";
  const hasMenu = action.files.length > 1;
  const button = (
    <Button
      appearance="overlay"
      hierarchy="secondary"
      size="md"
      isIconOnly
      leftIcon={action.kind === "reveal" ? <FolderOpen /> : <Download />}
      aria-label={label}
      onClick={hasMenu ? undefined : action.files[0]?.onSelect}
    />
  );

  return (
    <Tooltip content={label}>
      <span>
        {hasMenu ? (
          <DropdownMenu
            items={action.files.map((file) => ({
              render: <ItemAction text={file.label} onClick={file.onSelect} />,
            }))}
            align="end"
            shouldBeModal={false}
            portalContainer={rootElement}
          >
            {button}
          </DropdownMenu>
        ) : (
          button
        )}
      </span>
    </Tooltip>
  );
}
