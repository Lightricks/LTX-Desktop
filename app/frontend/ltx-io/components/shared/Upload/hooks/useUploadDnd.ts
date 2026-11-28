import { useMergedRef } from "@mantine/hooks";
import { useRef, useState } from "react";

type BaseUseUploadDndProps = {
  disabled?: boolean;
  acceptedFilesTypes?: string;
  externalInputRef?: React.Ref<HTMLInputElement>;
  clickDisabled?: boolean;
  /** Override click handler (e.g., to open ImportAssetModal instead of native file picker) */
  onClickOverride?: () => void;
};

type SingleUseUploadDndProps = BaseUseUploadDndProps & {
  isMultiple: false;
  onUpload: (file: File, sourceAction?: "drag" | "click") => void;
};

type MultipleUseUploadDndProps = BaseUseUploadDndProps & {
  isMultiple: true;
  onUpload: (files: File[], sourceAction?: "drag" | "click") => void;
};

export type UseUploadDndProps = SingleUseUploadDndProps | MultipleUseUploadDndProps;

export function useUploadDnd(props: UseUploadDndProps) {
  const {
    disabled = false,
    acceptedFilesTypes = "",
    externalInputRef,
    clickDisabled = false,
    onClickOverride,
  } = props;

  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const mergedRef = useMergedRef(inputRef, externalInputRef);

  const handleDragEnter = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(true);
  };

  const handleDragOver = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();

    // Determine if the drag is leaving the component by checking
    // if the related target is not a descendant of the current target.
    // This assumes that the component's children are part of the drag area.
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      if (props.isMultiple) {
        props.onUpload(files, "drag");
      } else {
        props.onUpload(files[0], "drag");
      }
    }
  };

  const handleInputClick = () => {
    if (disabled || clickDisabled) return;

    if (onClickOverride) {
      onClickOverride();
      return;
    }
    inputRef.current?.click();
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event?.target?.files ? Array.from(event.target.files) : [];
    if (files.length === 0) return;
    event.target.value = "";
    if (props.isMultiple) {
      props.onUpload(files, "click");
    } else {
      props.onUpload(files[0], "click");
    }
  };

  return {
    isDragOver,
    inputRef: mergedRef,
    handlers: {
      handleDragEnter,
      handleDragOver,
      handleDragLeave,
      handleDrop,
      handleInputClick,
      handleInputChange,
    },
    inputProps: {
      type: "file",
      accept: acceptedFilesTypes,
      disabled,
      multiple: props.isMultiple,
    },
  };
}
