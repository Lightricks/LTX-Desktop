import { useCallback, useEffect, useRef } from "react";
import { Navigate, useNavigate } from "react-router";

import { customLoraRunPath, paths } from "@/paths";

import { useModalContext } from "../../components/shared/Modal/ModalContext";
import { useCustomLoraImport } from "../../hooks/useCustomLoras";
import {
  type ExploreCustomLoraImport,
  useExploreRuntime,
} from "../../runtime/ExploreRuntime";

import { AddCustomLoraPage } from "./AddCustomLoraPage";
import styles from "./CustomLoraFeatureScreen.module.scss";

/**
 * Nothing in the app opens this path anymore. Saved LoRAs are cards on Home, in search, and in
 * the sidebar. Adding one is a dialog. A leftover visit goes back to Home.
 */
export function CustomLoraFeatureScreen() {
  return <Navigate to={paths.home} replace />;
}

function AddLora({
  files,
  onClose,
  onImported,
  runningFlag,
}: {
  files: ExploreCustomLoraImport;
  onClose: () => void;
  onImported: (customLoraId: string) => void;
  /** The dialog reads this at close time so a running import cannot be dismissed. */
  runningFlag: { current: boolean };
}) {
  const { api } = useExploreRuntime();
  const importer = useCustomLoraImport(api, onImported);
  useEffect(() => {
    runningFlag.current = importer.state.status === "running";
    return () => {
      runningFlag.current = false;
    };
  }, [importer.state.status, runningFlag]);
  const fetchCardSuggestions = useCallback(
    async (url: string) => {
      const result = await api.getCustomLoraLinkInfo({ url });
      return result.ok ? result.data : null;
    },
    [api],
  );
  const close = () => {
    if (importer.state.status === "running") return;
    onClose();
  };
  return (
    <AddCustomLoraPage
      importState={importer.state}
      files={files}
      fetchCardSuggestions={fetchCardSuggestions}
      onClose={close}
      onDismissError={importer.dismissError}
      onSubmit={(request) => {
        const { coverPath, ...body } = request;
        void importer.start(
          coverPath == null ? body : { ...body, cover_path: coverPath },
        );
      }}
    />
  );
}

/** Opens the add dialog over the page the user is already on. */
export function useAddCustomLoraModal() {
  const { customLoraImport } = useExploreRuntime();
  const { showModal, hideModal } = useModalContext();
  const navigate = useNavigate();
  const runningFlag = useRef(false);
  useEffect(() => {
    return () => {
      runningFlag.current = false;
    };
  }, []);
  return useCallback(() => {
    if (customLoraImport == null) return;
    runningFlag.current = false;
    showModal({
      content: (
        <AddLora
          files={customLoraImport}
          runningFlag={runningFlag}
          onClose={() => hideModal("close")}
          onImported={(customLoraId) => {
            hideModal("done");
            void navigate(customLoraRunPath(customLoraId));
          }}
        />
      ),
      modalName: "add-custom-lora",
      modalClassName: styles.modal,
      variant: "secondary",
      noPadding: true,
      shouldPreventDismiss: () => runningFlag.current,
    });
  }, [customLoraImport, hideModal, navigate, showModal]);
}
