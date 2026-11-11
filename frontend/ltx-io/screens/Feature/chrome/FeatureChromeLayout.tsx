import { useCallback, useMemo } from "react";
import { Outlet, useMatches, useOutletContext, useParams } from "react-router";

import { PageHeader } from "@/components/home/PageHeader";
import { CUSTOM_LORA_HOME_FEATURE } from "@/lib/custom-lora";
import {
  type DesktopHomeOutletContext,
  pageHeaderQueueControlFromOutlet,
  showHistoryNavFromOutlet,
} from "@/lib/desktop-home-outlet-context";
import { getHomeFeature } from "@/lib/home-features";

import { useModalContext } from "../../../components/shared/Modal/ModalContext";
import { useCustomLoras } from "../../../hooks/useCustomLoras";

import { FeatureBreadcrumbs } from "./FeatureBreadcrumbs";
import styles from "./FeatureChromeLayout.module.scss";
import { FEATURE_DETAILS_MODAL_NAME, FeatureDetailsModal } from "./FeatureDetailsModal";
import detailsStyles from "./FeatureDetailsModal.module.scss";
import { featureCrumbTrail } from "./featureChromeModel";
import { selectHomeFeatureId } from "./homeFeatureRouteHandle";

export function FeatureChromeLayout() {
  const featureId = selectHomeFeatureId(useMatches());
  const { customLoraId } = useParams();
  // Only the page of one saved custom LoRA has this param. Its breadcrumb and details are the LoRA's.
  const customLoras = useCustomLoras({ enabled: customLoraId !== undefined });
  const customLora = customLoras.data?.find((lora) => lora.id === customLoraId);
  const { showModal } = useModalContext();
  const outletContext = useOutletContext<DesktopHomeOutletContext | null | undefined>();
  const queueControl = pageHeaderQueueControlFromOutlet(outletContext);
  const showHistoryNav = showHistoryNavFromOutlet(outletContext);
  const usesPageHeader = queueControl != null;

  // A custom LoRA has no info dialog. Other features still open theirs from the "i" button.
  const hasDetails = featureId !== CUSTOM_LORA_HOME_FEATURE.id;

  const openFeatureDetails = useCallback(() => {
    if (featureId == null || featureId === CUSTOM_LORA_HOME_FEATURE.id) return;
    showModal({
      content: <FeatureDetailsModal featureId={featureId} />,
      modalName: FEATURE_DETAILS_MODAL_NAME,
      modalClassName: detailsStyles.modalFrame,
      variant: "secondary",
      noPadding: true,
    });
  }, [featureId, showModal]);

  const breadcrumbs = useMemo(() => {
    if (featureId == null) return null;
    const feature = getHomeFeature(featureId);
    return (
      <FeatureBreadcrumbs
        trail={featureCrumbTrail(feature, customLora?.name)}
        detailsLabel={`About ${customLora?.name ?? feature.title}`}
        variant={usesPageHeader ? "topBar" : "default"}
        onOpenDetails={hasDetails ? openFeatureDetails : undefined}
      />
    );
  }, [customLora, featureId, hasDetails, openFeatureDetails, usesPageHeader]);

  if (featureId == null) {
    return <Outlet />;
  }

  return (
    <div className={styles.chrome}>
      {usesPageHeader && breadcrumbs ? (
        <PageHeader
          insetFromShell
          actionsAtEndWhenWrapped
          showHistoryNav={showHistoryNav}
          leading={breadcrumbs}
          queueControl={queueControl}
          beforeActions={outletContext?.beforeHeaderActions}
        />
      ) : (
        breadcrumbs
      )}
      <div className={styles.outlet}>
        <Outlet />
      </div>
    </div>
  );
}
