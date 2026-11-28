import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@ds/DropdownMenu/DropdownMenu";
import type { MenuItem } from "@ds/DropdownMenu/DropdownMenuItem";
import AudioIcon from "@ds/assets/Icons/Audio/On.svg?react";
import RecordIcon from "@ds/assets/Icons/Record.svg?react";
import { useThemeRootElement } from "@ds/styles/themes/useTheme";
import type { ReactElement } from "react";

export function AudioSourceMenu({
  enabled,
  items,
  children,
}: {
  enabled: boolean;
  items: MenuItem[];
  children: ReactElement;
}) {
  const rootElement = useThemeRootElement();

  if (!enabled) {
    return children;
  }

  return (
    <DropdownMenu
      items={items}
      align="start"
      sideOffset={4}
      minWidth="160px"
      shouldBeModal={false}
      portalContainer={rootElement}
    >
      {children}
    </DropdownMenu>
  );
}

export function buildAudioSourceMenuItems({
  onImport,
  onRecord,
  importLabel,
}: {
  onImport: () => void;
  onRecord?: () => void;
  importLabel: string;
}): MenuItem[] {
  const items: MenuItem[] = [
    {
      render: (
        <ItemAction
          text={importLabel}
          leftIcon={<AudioIcon />}
          onClick={onImport}
        />
      ),
    },
  ];
  if (onRecord != null) {
    items.push({
      render: (
        <ItemAction
          text="Record audio"
          leftIcon={<RecordIcon />}
          onClick={onRecord}
        />
      ),
    });
  }
  return items;
}
