import { Button } from "@ds/Button/Button";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownPicker } from "@ds/DropdownPicker/DropdownPicker";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import ArrowDownIcon from "@ds/assets/Icons/Arrow/Down/Small.svg?react";
import AudioIcon from "@ds/assets/Icons/Audio/On.svg?react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";
import PhotoIcon from "@ds/assets/Icons/Photo.svg?react";
import VideoCameraIcon from "@ds/assets/Icons/VideoCamera.svg?react";
import { useTheme } from "@ds/styles/themes/useTheme";
import { ArrowDownWideNarrow, ArrowUpNarrowWide, LayoutGrid, Search } from "lucide-react";
import { type ReactNode, useState } from "react";
import { useOutletContext } from "react-router";

import { PageHeader } from "@/components/home/PageHeader";
import {
  type DesktopHomeOutletContext,
  pageHeaderQueueControlFromOutlet,
  showHistoryNavFromOutlet,
} from "@/lib/desktop-home-outlet-context";

import styles from "./AssetsLibraryScreen.module.scss";
import { type AssetsListFilters, type AssetsMediaKind } from "./assetsSearchParams";

const TYPE_OPTIONS = [
  { value: "all", label: "All media" },
  { value: "image", label: "Images" },
  { value: "video", label: "Videos" },
  { value: "audio", label: "Audio" },
] as const;

const SORT_OPTIONS = [
  { value: "created_at-desc", label: "Newest" },
  { value: "created_at-asc", label: "Oldest" },
] as const;

const PICKER_CONTENT_PROPS = {
  side: "bottom" as const,
  align: "start" as const,
  sideOffset: 8,
  avoidCollisions: true,
  collisionPadding: 8,
  style: {
    zIndex: "calc(var(--z-modal) + 1)",
  },
};

type TypeFilterValue = (typeof TYPE_OPTIONS)[number]["value"];
type SortFilterValue = (typeof SORT_OPTIONS)[number]["value"];

function mediaKindToTypeValue(mediaKind: AssetsMediaKind | null): TypeFilterValue {
  return mediaKind ?? "all";
}

function typeValueToMediaKind(value: TypeFilterValue): AssetsMediaKind | null {
  switch (value) {
    case "all":
      return null;
    case "image":
    case "video":
    case "audio":
      return value;
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}

function parseTypeValue(value: string): TypeFilterValue {
  switch (value) {
    case "all":
    case "image":
    case "video":
    case "audio":
      return value;
    default:
      return "all";
  }
}

function parseSortValue(value: string): SortFilterValue {
  switch (value) {
    case "created_at-desc":
    case "created_at-asc":
      return value;
    default:
      return "created_at-desc";
  }
}

function typeFilterIcon(value: TypeFilterValue): ReactNode {
  switch (value) {
    case "all":
      return <LayoutGrid aria-hidden />;
    case "image":
      return <PhotoIcon aria-hidden />;
    case "video":
      return <VideoCameraIcon aria-hidden />;
    case "audio":
      return <AudioIcon aria-hidden />;
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}

function sortDirectionIcon(value: SortFilterValue): ReactNode {
  switch (value) {
    case "created_at-desc":
      return <ArrowDownWideNarrow aria-hidden />;
    case "created_at-asc":
      return <ArrowUpNarrowWide aria-hidden />;
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}

function ToolbarFilterButton({
  label,
  value,
  valueLabel,
  leftIcon,
  options,
  contentProps = PICKER_CONTENT_PROPS,
  onValueChange,
}: {
  label: string;
  value: string;
  valueLabel: string;
  leftIcon: ReactNode;
  options: readonly { value: string; label: string; icon: ReactNode }[];
  contentProps?: typeof PICKER_CONTENT_PROPS;
  onValueChange: (value: string) => void;
}) {
  const { rootElement } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <DropdownPicker
      options={options.map((option) => ({
        value: option.value,
        label: option.label,
        render: (
          <ItemAction
            text={option.label}
            textVariant="body"
            leftIcon={option.icon}
            selectionPosition="right"
            isSelectable
            isSelected={value === option.value}
          />
        ),
      }))}
      value={value}
      label={label}
      isOpen={isOpen}
      onOpenChange={setIsOpen}
      onValueChange={onValueChange}
      portalContainer={rootElement}
      contentProps={contentProps}
      minWidth="160px"
    >
      <Button
        appearance="neutral"
        hierarchy="plain"
        size="lg"
        aria-expanded={isOpen}
        label={valueLabel}
        leftIcon={leftIcon}
        rightIcon={<ArrowDownIcon />}
      />
    </DropdownPicker>
  );
}

function AssetsSearchField({
  value,
  onValueChange,
}: {
  value: string;
  onValueChange: (value: string) => void;
}) {
  const hasValue = value.length > 0;

  return (
    <div className={styles.searchFieldContainer}>
      <Search className={styles.searchIcon} size={12} strokeWidth={2} aria-hidden />
      <label htmlFor="assets-search" className={styles.visuallyHidden}>
        Search
      </label>
      <input
        id="assets-search"
        className={styles.searchField}
        data-ltxio-type="body-md"
        placeholder="Search"
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        autoComplete="off"
        data-1p-ignore
      />
      {hasValue ? (
        <Button
          className={styles.searchClearButton}
          appearance="neutral"
          hierarchy="plain"
          size="sm"
          isIconOnly
          aria-label="Clear search"
          leftIcon={<CloseIcon />}
          onClick={() => onValueChange("")}
        />
      ) : null}
    </div>
  );
}

export function AssetsToolbar({
  filters,
  searchDraft,
  onSearchDraftChange,
  onFiltersChange,
}: {
  filters: AssetsListFilters;
  searchDraft: string;
  onSearchDraftChange: (value: string) => void;
  onFiltersChange: (filters: AssetsListFilters) => void;
}) {
  const outletContext = useOutletContext<DesktopHomeOutletContext | null | undefined>();
  const queueControl = pageHeaderQueueControlFromOutlet(outletContext);
  const showHistoryNav = showHistoryNavFromOutlet(outletContext);
  const typeValue = mediaKindToTypeValue(filters.mediaKind);
  const typeLabel =
    TYPE_OPTIONS.find((option) => option.value === typeValue)?.label ?? "All media";
  const sortLabel =
    SORT_OPTIONS.find((option) => option.value === filters.sort)?.label ?? "Newest";

  const filterControls = (
    <div className={styles.filters}>
      <Tooltip content="Filter by media type">
        <span className={styles.tooltipAnchor}>
          <ToolbarFilterButton
            label="Filter by media type"
            value={typeValue}
            valueLabel={typeLabel}
            leftIcon={typeFilterIcon(typeValue)}
            options={TYPE_OPTIONS.map((option) => ({
              value: option.value,
              label: option.label,
              icon: typeFilterIcon(option.value),
            }))}
            onValueChange={(next) => {
              onFiltersChange({
                ...filters,
                mediaKind: typeValueToMediaKind(parseTypeValue(next)),
              });
            }}
          />
        </span>
      </Tooltip>
      <Tooltip content="Sort by date added">
        <span className={styles.tooltipAnchor}>
          <ToolbarFilterButton
            label="Sort by date added"
            value={filters.sort}
            valueLabel={sortLabel}
            leftIcon={sortDirectionIcon(filters.sort)}
            options={SORT_OPTIONS.map((option) => ({
              value: option.value,
              label: option.label,
              icon: sortDirectionIcon(option.value),
            }))}
            onValueChange={(next) => {
              onFiltersChange({
                ...filters,
                sort: parseSortValue(next),
              });
            }}
          />
        </span>
      </Tooltip>
    </div>
  );

  if (queueControl != null) {
    return (
      <PageHeader
        insetFromShell
        showHistoryNav={showHistoryNav}
        leading={filterControls}
        beforeActions={
          <AssetsSearchField value={searchDraft} onValueChange={onSearchDraftChange} />
        }
        queueControl={queueControl}
      />
    );
  }

  return (
    <div className={styles.toolbar}>
      {filterControls}
      <AssetsSearchField value={searchDraft} onValueChange={onSearchDraftChange} />
    </div>
  );
}
