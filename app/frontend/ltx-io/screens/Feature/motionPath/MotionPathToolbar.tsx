import { Button } from "@ds/Button/Button";
import { ShortcutTooltipContent } from "@ds/ShortcutTooltipContent/ShortcutTooltipContent";
import BrushIcon from "@ds/assets/Icons/Brush.svg?react";
import ConvertPointIcon from "@ds/assets/Icons/ConvertPoint.svg?react";
import PenIcon from "@ds/assets/Icons/Pen.svg?react";
import PenEraseIcon from "@ds/assets/Icons/PenErase.svg?react";
import RedoIcon from "@ds/assets/Icons/Redo.svg?react";
import UndoIcon from "@ds/assets/Icons/Undo.svg?react";

import type { MotionPathTool } from "./MotionPathCanvas";
import { MotionPathToolbarIconButton } from "./MotionPathToolbarIconButton";
import toolbarStyles from "./MotionPathToolbar.module.scss";

const IS_MAC = /mac/i.test(navigator.platform);
const COMMAND_ERASE_SHORTCUT_DISPLAY = IS_MAC ? "⌘" : "Ctrl";
const OPTION_CONVERT_SHORTCUT_DISPLAY = IS_MAC ? "⌥" : "Alt";
const UNDO_SHORTCUT_DISPLAY = IS_MAC ? "⌘Z" : "Ctrl+Z";
const REDO_SHORTCUT_DISPLAY = IS_MAC ? "⌘⇧Z" : "Ctrl+Shift+Z";

export type MotionPathToolbarProps = {
  activeTool: MotionPathTool;
  onSelectTool: (tool: MotionPathTool) => void;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
  canUndo: boolean;
  canRedo: boolean;
  canClear: boolean;
};

export function MotionPathToolbar({
  activeTool,
  onSelectTool,
  onUndo,
  onRedo,
  onClear,
  canUndo,
  canRedo,
  canClear,
}: MotionPathToolbarProps) {
  return (
    <div className={toolbarStyles.toolbar}>
      <MotionPathToolbarIconButton
        tooltip={<ShortcutTooltipContent title="Pen" shortcut="P" />}
        icon={<PenIcon />}
        label="Pen"
        pressed={activeTool === "pen"}
        onClick={() => onSelectTool("pen")}
      />
      <MotionPathToolbarIconButton
        tooltip={
          <ShortcutTooltipContent
            title="Remove points"
            shortcut={COMMAND_ERASE_SHORTCUT_DISPLAY}
          />
        }
        icon={<PenEraseIcon />}
        label="Remove points"
        pressed={activeTool === "erase"}
        onClick={() => onSelectTool("erase")}
      />
      <MotionPathToolbarIconButton
        tooltip={
          <ShortcutTooltipContent
            title="Bend"
            shortcut={OPTION_CONVERT_SHORTCUT_DISPLAY}
          />
        }
        icon={<ConvertPointIcon />}
        label="Bend"
        pressed={activeTool === "convert"}
        onClick={() => onSelectTool("convert")}
      />
      <MotionPathToolbarIconButton
        tooltip={<ShortcutTooltipContent title="Free draw" shortcut="B" />}
        icon={<BrushIcon />}
        label="Free draw"
        pressed={activeTool === "freehand"}
        onClick={() => onSelectTool("freehand")}
      />
      <div className={toolbarStyles.separator} />
      <MotionPathToolbarIconButton
        tooltip={
          <ShortcutTooltipContent title="Undo" shortcut={UNDO_SHORTCUT_DISPLAY} />
        }
        icon={<UndoIcon />}
        label="Undo"
        disabled={!canUndo}
        onClick={onUndo}
      />
      <MotionPathToolbarIconButton
        tooltip={
          <ShortcutTooltipContent title="Redo" shortcut={REDO_SHORTCUT_DISPLAY} />
        }
        icon={<RedoIcon />}
        label="Redo"
        disabled={!canRedo}
        onClick={onRedo}
      />
      <Button
        appearance="neutral"
        hierarchy="plain"
        size="md"
        label="Clear"
        className={toolbarStyles.clearButton}
        disabled={!canClear}
        onClick={onClear}
      />
    </div>
  );
}
