import type { IcLoraListItem, LoraCatalogListItem } from "../../../hooks/use-catalog.ts";
import { catalogDefaultRepoId } from "../../../lib/lora-library.ts";
import {
  catalogVariantsFor,
  type CatalogVariantChoice,
} from "../../lib/catalogVariantChoice.ts";

export type ListedEntry = {
  id: string;
  downloaded: boolean;
  requiresHfLogin: boolean;
  repoId: string | undefined;
  sizeBytes: number | undefined;
  defaultLoraStrength: number | undefined;
  defaultAudioMode: "source" | "generated" | "off" | undefined;
  variants: CatalogVariantChoice[];
};

export function listedIcLoraEntry(entry: IcLoraListItem): ListedEntry {
  return {
    id: entry.ic_lora.id,
    downloaded: entry.downloaded,
    requiresHfLogin: entry.ic_lora.requires_hf_login,
    repoId: catalogDefaultRepoId(entry.ic_lora.download),
    sizeBytes: entry.ic_lora.download.variants[0]?.size_bytes,
    defaultLoraStrength: entry.ic_lora.default_settings?.lora_strength,
    defaultAudioMode: entry.ic_lora.default_settings?.audio_mode ?? undefined,
    variants: catalogVariantsFor(entry.ic_lora, entry.downloaded_variant_ids),
  };
}

export function listedLoraEntry(entry: LoraCatalogListItem): ListedEntry {
  return {
    id: entry.lora.id,
    downloaded: entry.downloaded,
    requiresHfLogin: entry.lora.requires_hf_login,
    repoId: catalogDefaultRepoId(entry.lora.download),
    sizeBytes: entry.lora.download.variants[0]?.size_bytes,
    defaultLoraStrength: undefined,
    defaultAudioMode: undefined,
    variants: catalogVariantsFor(entry.lora, entry.downloaded_variant_ids),
  };
}
