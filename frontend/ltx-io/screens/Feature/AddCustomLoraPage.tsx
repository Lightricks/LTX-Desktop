import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { TextField } from "@ds/TextField/TextField";
import { Tooltip, TooltipProvider } from "@ds/Tooltip/Tooltip";
import CircleCheckIcon from "@ds/assets/Icons/CircleCheck.svg?react";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { clsx } from "clsx";
import {
  FileText,
  Image as ImageIcon,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  CUSTOM_LORA_DESCRIPTION_MAX_LENGTH,
  CUSTOM_LORA_EXAMPLES_MAX,
  CUSTOM_LORA_EXAMPLE_MAX_LENGTH,
  CUSTOM_LORA_NAME_MAX_LENGTH,
  CUSTOM_LORA_TRIGGER_MAX_LENGTH,
} from "@/lib/custom-lora";
import { pathToFileUrl } from "@/lib/file-url";
import { formatBytes } from "@/lib/format";

import {
  DEFAULT_AUTOCLOSE_MS,
  createToast,
} from "../../components/shared/Toast/toastService";
import type { CustomLoraImportState } from "../../hooks/useCustomLoras";
import type { ExploreCustomLoraImport } from "../../runtime/ExploreRuntime";

import styles from "./AddCustomLoraPage.module.scss";
import { CustomLoraDropzone } from "./CustomLoraDropzone";
import {
  type CustomLoraCardSuggestions,
  type CustomLoraInfo,
  EMPTY_CUSTOM_LORA_INFO,
  applyCardSuggestions,
  customLoraInfoRequest,
  importErrorField,
} from "./customLoraInfo";
import { prepareDroppedImageIngest } from "./fields/imageAssetInput";

export type CustomLoraImportSubmit = {
  name: string;
  description?: string;
  trigger?: string;
  examples: string[];
  /** A local image to show on the Home card. Left out when the user skips the cover. */
  coverPath?: string;
} & ({ source_path: string } | { url: string });

// Wait for the user to stop typing before the page reads the model card.
const CARD_LOOKUP_DELAY_MS = 600;

function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] ?? path;
}

function isSafetensorsPath(path: string): boolean {
  return path.toLowerCase().endsWith(".safetensors");
}

const UNSUPPORTED_WEIGHTS_MESSAGE = "Choose a .safetensors file.";

const CUSTOM_LORA_DESCRIPTION_HINT =
  "The more details you provide, the better the prompt enhancer can tailor prompts to your LoRA.";

const CUSTOM_LORA_LINK_NOTE = "Hugging Face model page or file link.";

const CUSTOM_LORA_INVALID_LINK_NOTE =
  "Invalid link. Use only Hugging Face model page or file link.";

/**
 * The page where the user adds a LoRA: a name, the type, and the weights as a link or a file.
 * When the import ends, the parent opens the page of the new LoRA.
 */
export function AddCustomLoraPage({
  importState,
  files,
  fetchCardSuggestions,
  onSubmit,
  onClose,
  onDismissError,
}: {
  importState: CustomLoraImportState;
  files: ExploreCustomLoraImport;
  /** Reads the Hugging Face model card behind a link. Gives `null` when it cannot. */
  fetchCardSuggestions: (url: string) => Promise<CustomLoraCardSuggestions | null>;
  onSubmit: (request: CustomLoraImportSubmit) => void;
  /** Closes the dialog. */
  onClose: () => void;
  /** Clears the last import error once the user edits the name, the file, or the link. */
  onDismissError?: () => void;
}) {
  const { pickWeightsFile, pickCoverFile, pathForDroppedFile, fileSize } = files;
  const [name, setName] = useState("");
  const [filePath, setFilePath] = useState<string | null>(null);
  const [fileSizeBytes, setFileSizeBytes] = useState<number | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [checkedLink, setCheckedLink] = useState<{
    url: string;
    accepted: boolean;
  } | null>(null);
  const [info, setInfo] = useState<CustomLoraInfo>(EMPTY_CUSTOM_LORA_INFO);
  // Remembered with the LoRA so Home can show this picture instead of the shared clip.
  const [coverPath, setCoverPath] = useState<string | null>(null);
  const [coverError, setCoverError] = useState<string | null>(null);
  const [submitSource, setSubmitSource] = useState<"file" | "link" | null>(null);
  const importErrorRef = useRef<HTMLDivElement>(null);
  const importMessage = importState.status === "error" ? importState.message : null;
  const importCode = importState.status === "error" ? importState.code : undefined;
  const importField =
    importMessage == null ? null : importErrorField(importCode, submitSource);

  useEffect(() => {
    if (importState.status !== "error") return;
    createToast({
      message: importState.message,
      toastType: "danger",
      closeable: true,
      autoCloseAfter: DEFAULT_AUTOCLOSE_MS,
    });
    importErrorRef.current?.scrollIntoView({ block: "nearest" });
  }, [importState]);

  const running = importState.status === "running";
  const trimmedName = name.trim();
  const trimmedUrl = url.trim();
  const linkFilled = trimmedUrl.length > 0;
  const linkAccepted = checkedLink?.url === trimmedUrl && checkedLink.accepted;
  const hasValidSource = filePath != null || linkAccepted;
  const canSubmit = !running && trimmedName.length > 0 && hasValidSource;
  // A link and a file are alternatives. Whichever one is filled turns the other off.
  const linkInvalid = checkedLink?.url === trimmedUrl && !checkedLink.accepted;
  const fileChosen = filePath != null;
  const dropDisabled = running || linkFilled;

  // The backend checks the link and reads any Hugging Face model card behind it.
  useEffect(() => {
    if (!linkFilled) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void fetchCardSuggestions(trimmedUrl).then((suggestions) => {
        if (cancelled) return;
        const accepted = suggestions != null && suggestions.accepted !== false;
        setCheckedLink({ url: trimmedUrl, accepted });
        if (accepted && suggestions != null) {
          setInfo((current) => applyCardSuggestions(current, suggestions));
        }
      });
    }, CARD_LOOKUP_DELAY_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [linkFilled, trimmedUrl, fetchCardSuggestions]);

  const setExample = (index: number, text: string) =>
    setInfo((current) => ({
      ...current,
      examples: current.examples.map((example, i) => (i === index ? text : example)),
    }));

  const removeExample = (index: number) =>
    setInfo((current) => ({
      ...current,
      examples:
        current.examples.length < 2
          ? current.examples
          : current.examples.filter((_, i) => i !== index),
    }));

  const acceptWeightsPath = (path: string, sizeBytes: number | null) => {
    onDismissError?.();
    if (!isSafetensorsPath(path)) {
      setFileError(UNSUPPORTED_WEIGHTS_MESSAGE);
      return;
    }
    setFileError(null);
    setFilePath(path);
    setFileSizeBytes(sizeBytes);
    setUrl("");
  };

  const clearFile = () => {
    onDismissError?.();
    setFilePath(null);
    setFileSizeBytes(null);
    setFileError(null);
  };

  const acceptCoverPath = (path: string) => {
    onDismissError?.();
    const dropped = prepareDroppedImageIngest(
      { name: fileNameOf(path), type: "" },
      () => path,
    );
    if ("error" in dropped) {
      setCoverError(dropped.error);
      return;
    }
    setCoverError(null);
    setCoverPath(dropped.ingestPath);
  };

  const clearCover = () => {
    onDismissError?.();
    setCoverPath(null);
    setCoverError(null);
  };

  const chooseCover = () => {
    if (running) return;
    void pickCoverFile().then((picked) => {
      if (picked == null) return;
      acceptCoverPath(picked);
    });
  };

  const onCoverDrop = (file: File) => {
    onDismissError?.();
    const dropped = prepareDroppedImageIngest(file, () => pathForDroppedFile(file));
    if ("error" in dropped) {
      setCoverError(dropped.error);
      return;
    }
    setCoverError(null);
    setCoverPath(dropped.ingestPath);
  };

  const chooseFile = () => {
    if (running || linkFilled) return;
    void pickWeightsFile().then((picked) => {
      if (picked == null) return;
      void fileSize(picked).then((sizeBytes) => {
        acceptWeightsPath(picked, sizeBytes);
      });
    });
  };

  const onWeightsDrop = (file: File) => {
    const path = pathForDroppedFile(file);
    if (path == null) {
      setFileError(UNSUPPORTED_WEIGHTS_MESSAGE);
      return;
    }
    acceptWeightsPath(path, file.size);
  };

  const submit = () => {
    if (!canSubmit) return;
    const details = customLoraInfoRequest(info);
    const cover = coverPath == null ? {} : { coverPath };
    setSubmitSource(filePath != null ? "file" : "link");
    if (filePath != null) {
      onSubmit({ name: trimmedName, source_path: filePath, ...details, ...cover });
      return;
    }
    onSubmit({ name: trimmedName, url: trimmedUrl, ...details, ...cover });
  };

  const importError =
    importMessage == null ? null : (
      <div ref={importErrorRef}>
        <Text as="p" variant="body" size="sm" className={styles.error}>
          {importMessage}
        </Text>
      </div>
    );

  return (
    <div className={styles.page}>
      <div className={styles.panel}>
        <div className={styles.fields}>
          <div className={styles.titleRow}>
            <Text as="h1" variant="heading" size="md" className={styles.title}>
              Add a custom LoRA
            </Text>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              isIconOnly
              aria-label="Close"
              leftIcon={<X />}
              disabled={running}
              onClick={onClose}
            />
          </div>
          <div className={styles.field}>
            <Text as="span" variant="body" size="md" className={styles.label}>
              Name
              <span className={styles.required} aria-hidden="true">
                *
              </span>
            </Text>
            <TextField
              aria-label="LoRA name"
              placeholder="For example, Ink wash style"
              value={name}
              maxLength={CUSTOM_LORA_NAME_MAX_LENGTH}
              disabled={running}
              autoFocus
              className={styles.input}
              onInputChange={(text) => {
                setName(text);
                onDismissError?.();
              }}
            />
            {importField === "name" ? importError : null}
          </div>

          <div className={styles.field}>
            <Text as="span" variant="body" size="md" className={styles.label}>
              Add LoRA file
              <span className={styles.required} aria-hidden="true">
                *
              </span>
            </Text>
            {filePath != null ? (
              <div className={styles.uploadedFile}>
                <span className={styles.fileTile} aria-hidden>
                  <FileText className={styles.fileTileIcon} />
                </span>
                <div className={styles.uploadedMeta}>
                  <Text
                    as="span"
                    variant="body"
                    size="md"
                    className={styles.uploadedName}
                    title={filePath}
                  >
                    {fileNameOf(filePath)}
                  </Text>
                  {fileSizeBytes != null ? (
                    <Text
                      as="span"
                      variant="body"
                      size="sm"
                      className={styles.uploadedSize}
                    >
                      {formatBytes(fileSizeBytes)}
                    </Text>
                  ) : null}
                </div>
                <TooltipProvider delay={70}>
                  <div className={styles.uploadedActions}>
                    <Tooltip content="Remove">
                      <Button
                        appearance="neutral"
                        hierarchy="plain"
                        size="md"
                        isIconOnly
                        aria-label="Remove"
                        leftIcon={<Trash2 />}
                        disabled={running}
                        onClick={clearFile}
                      />
                    </Tooltip>
                    <Tooltip content="Replace">
                      <Button
                        appearance="neutral"
                        hierarchy="plain"
                        size="md"
                        isIconOnly
                        aria-label="Replace"
                        leftIcon={<RefreshCw />}
                        disabled={running}
                        onClick={chooseFile}
                      />
                    </Tooltip>
                  </div>
                </TooltipProvider>
              </div>
            ) : (
              <CustomLoraDropzone
                disabled={dropDisabled}
                label={
                  linkFilled
                    ? "File drop is off while a link is entered"
                    : "Drop a .safetensors file, or click to browse"
                }
                onActivate={chooseFile}
                onDropFile={onWeightsDrop}
              >
                <div className={styles.dropBody}>
                  <Upload className={styles.dropIcon} aria-hidden />
                  <Text as="span" variant="body" size="md" className={styles.fileName}>
                    Drop a .safetensors file here
                  </Text>
                  <Text as="span" variant="body" size="sm" className={styles.dropHint}>
                    or click to browse
                  </Text>
                </div>
              </CustomLoraDropzone>
            )}
            {fileError != null && fileError !== importMessage ? (
              <Text as="p" variant="body" size="sm" className={styles.error}>
                {fileError}
              </Text>
            ) : null}
            {importField === "file" ? importError : null}
            <div className={styles.or} role="separator">
              <span className={styles.orLine} />
              <Text as="span" variant="body" size="lg" className={styles.orLabel}>
                or
              </Text>
              <span className={styles.orLine} />
            </div>
            <div className={styles.linkBlock}>
              <Text as="span" variant="body" size="md" className={styles.label}>
                Add link
              </Text>
              <div className={styles.linkField}>
                <TextField
                  aria-label="LoRA weights link"
                  placeholder="https://huggingface.co/owner/model"
                  value={url}
                  disabled={running || fileChosen}
                  className={clsx(styles.input, linkAccepted && styles.inputWithStatus)}
                  onInputChange={(text) => {
                    onDismissError?.();
                    setUrl(text);
                    if (text.trim() !== "") {
                      setFilePath(null);
                      setFileSizeBytes(null);
                      setFileError(null);
                    }
                  }}
                />
                {linkAccepted ? (
                  <CircleCheckIcon className={styles.validIcon} aria-label="Valid link" />
                ) : null}
              </div>
              <Text
                as="p"
                variant="body"
                size="sm"
                className={clsx(
                  linkInvalid ? styles.error : styles.hint,
                  !linkInvalid && fileChosen && styles.hintDisabled,
                )}
              >
                {linkInvalid ? CUSTOM_LORA_INVALID_LINK_NOTE : CUSTOM_LORA_LINK_NOTE}
              </Text>
              {importField === "link" ? importError : null}
            </div>
          </div>

          <section className={styles.section}>
            <div className={styles.field}>
              <Text as="span" variant="body" size="md" className={styles.label}>
                LoRA description
              </Text>
              <textarea
                className={styles.textarea}
                aria-label="LoRA description"
                placeholder="For example, a watercolor ink wash style with soft edges and muted colors."
                value={info.description}
                maxLength={CUSTOM_LORA_DESCRIPTION_MAX_LENGTH}
                disabled={running}
                rows={3}
                onChange={(event) =>
                  setInfo((current) => ({ ...current, description: event.target.value }))
                }
              />
              <Text as="p" variant="body" size="sm" className={styles.hint}>
                {CUSTOM_LORA_DESCRIPTION_HINT}
              </Text>
            </div>

            <div className={styles.field}>
              <Text as="span" variant="body" size="md" className={styles.label}>
                Trigger word
              </Text>
              <TextField
                aria-label="Trigger word"
                placeholder="For example, InkW4sh"
                value={info.trigger}
                maxLength={CUSTOM_LORA_TRIGGER_MAX_LENGTH}
                disabled={running}
                className={styles.input}
                onInputChange={(text) =>
                  setInfo((current) => ({ ...current, trigger: text }))
                }
              />
            </div>

            <div className={styles.field}>
              <div className={styles.exampleTitle}>
                <Text as="span" variant="body" size="md" className={styles.label}>
                  Example prompt
                </Text>
                {info.examples.length < CUSTOM_LORA_EXAMPLES_MAX ? (
                  <TooltipProvider delay={70}>
                    <Tooltip content="Add another example">
                      <Button
                        appearance="neutral"
                        hierarchy="plain"
                        size="md"
                        isIconOnly
                        aria-label="Add another example"
                        leftIcon={<Plus />}
                        disabled={running}
                        onClick={() =>
                          setInfo((current) => ({
                            ...current,
                            examples: [...current.examples, ""],
                          }))
                        }
                      />
                    </Tooltip>
                  </TooltipProvider>
                ) : null}
              </div>
              {info.examples.map((example, index) => (
                <div key={index} className={styles.exampleRow}>
                  <textarea
                    // The list has no ids. An example field keeps its place, so the index is stable.
                    className={clsx(
                      styles.textarea,
                      index > 0 && styles.textareaWithRemove,
                    )}
                    aria-label={`Example prompt ${index + 1}`}
                    placeholder="For example, a felt fox walking through a paper forest"
                    value={example}
                    maxLength={CUSTOM_LORA_EXAMPLE_MAX_LENGTH}
                    disabled={running}
                    rows={3}
                    onChange={(event) => setExample(index, event.target.value)}
                  />
                  {index > 0 ? (
                    <div className={styles.exampleRemove}>
                      <TooltipProvider delay={70}>
                        <Tooltip content="Remove">
                          <Button
                            appearance="neutral"
                            hierarchy="plain"
                            size="xs"
                            isIconOnly
                            aria-label="Remove example"
                            leftIcon={<X />}
                            disabled={running}
                            onClick={() => removeExample(index)}
                          />
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </section>

          <div className={styles.field}>
            <Text as="span" variant="body" size="md" className={styles.label}>
              Cover image
            </Text>
            {coverPath != null ? (
              <div className={styles.coverFrame}>
                <img
                  className={styles.coverImage}
                  src={pathToFileUrl(coverPath)}
                  alt={fileNameOf(coverPath)}
                />
                <div className={styles.coverActions}>
                  <Tooltip content="Replace">
                    <Button
                      appearance="overlay"
                      hierarchy="secondary"
                      size="md"
                      isIconOnly
                      aria-label="Replace cover"
                      leftIcon={<MediaReplaceIcon />}
                      disabled={running}
                      onClick={chooseCover}
                    />
                  </Tooltip>
                  <Tooltip content="Clear">
                    <Button
                      appearance="overlay"
                      hierarchy="secondary"
                      size="md"
                      isIconOnly
                      aria-label="Clear cover"
                      leftIcon={<RemoveIcon />}
                      disabled={running}
                      onClick={clearCover}
                    />
                  </Tooltip>
                </div>
              </div>
            ) : (
              <CustomLoraDropzone
                disabled={running}
                className={styles.coverDrop}
                label="Drop a cover image, or click to browse"
                onActivate={chooseCover}
                onDropFile={onCoverDrop}
              >
                <div className={styles.dropBody}>
                  <ImageIcon className={styles.dropIcon} aria-hidden />
                  <Text as="span" variant="body" size="md" className={styles.fileName}>
                    Drop a cover image here
                  </Text>
                  <Text as="span" variant="body" size="sm" className={styles.dropHint}>
                    or click to browse
                  </Text>
                </div>
              </CustomLoraDropzone>
            )}
            {coverError != null ? (
              <Text as="p" variant="body" size="sm" className={styles.error}>
                {coverError}
              </Text>
            ) : null}
            {importField === "cover" ? importError : null}
          </div>

          {importField === "form" ? importError : null}
        </div>

        <div className={styles.actions}>
          <Button
            appearance="neutral"
            hierarchy="primary"
            size="lg"
            label={
              importState.status === "running"
                ? `Adding… ${Math.round(importState.progress)}%`
                : "Add LoRA"
            }
            isLoading={running}
            disabled={!canSubmit}
            className={styles.addButton}
            onClick={submit}
          />
        </div>
      </div>
    </div>
  );
}
