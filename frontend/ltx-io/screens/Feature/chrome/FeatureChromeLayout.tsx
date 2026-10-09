import { useCallback, useMemo } from "react";
import { Outlet, useMatches, useOutletContext } from "react-router";

import { PageHeader } from "@/components/home/PageHeader";
import {
  type DesktopHomeOutletContext,
  pageHeaderQueueControlFromOutlet,
  showHistoryNavFromOutlet,
} from "@/lib/desktop-home-outlet-context";
import { getHomeFeature } from "@/lib/home-features";

import { useModalContext } from "../../../components/shared/Modal/ModalContext";

import { FeatureBreadcrumbs } from "./FeatureBreadcrumbs";
import styles from "./FeatureChromeLayout.module.scss";
import { FEATURE_DETAILS_MODAL_NAME, FeatureDetailsModal } from "./FeatureDetailsModal";
import detailsStyles from "./FeatureDetailsModal.module.scss";
import { featureCrumbTrail } from "./featureChromeModel";
import { selectHomeFeatureId } from "./homeFeatureRouteHandle";

export function FeatureChromeLayout() {
  const featureId = selectHomeFeatureId(useMatches());
  const { showModal } = useModalContext();
  const outletContext = useOutletContext<DesktopHomeOutletContext | null | undefined>();
  const queueControl = pageHeaderQueueControlFromOutlet(outletContext);
  const showHistoryNav = showHistoryNavFromOutlet(outletContext);
  const usesPageHeader = queueControl != null;

  const openFeatureDetails = useCallback(() => {
    if (featureId == null) return;
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
        trail={featureCrumbTrail(feature)}
        detailsLabel={`About ${feature.title}`}
        variant={usesPageHeader ? "topBar" : "default"}
        onOpenDetails={openFeatureDetails}
      />
    );
  }, [featureId, openFeatureDetails, usesPageHeader]);

  if (featureId == null) {
    return <Outlet />;
  }

  return (
    <div className={styles.chrome}>
      {usesPageHeader && breadcrumbs ? (
        <PageHeader
          insetFromShell
          showHistoryNav={showHistoryNav}
          leading={breadcrumbs}
          queueControl={queueControl}
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
