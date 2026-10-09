import type { MouseEvent } from "react";
import { useReducedMotion } from "framer-motion";

import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import ExternalLinkIcon from "@ds/assets/Icons/ExternalLink.svg?react";
import LtxLogo from "@/assets/home/icons/LTX-logo.svg?react";
import {
  openExternalBrowserUrl,
  shouldInterceptHomeExternalUrl,
} from "@/components/home/home-external-links";
import type { HomeFeatureId } from "@/lib/home-features";
import { RoundCloseButton } from "../../../components/shared/RoundCloseButton/RoundCloseButton";
import { VideoPlayer } from "../../../components/VideoView/VideoPlayer";
import { useModalContext } from "../../../components/shared/Modal/ModalContext";

import {
  COMMUNITY_FEATURE_DISCLAIMER,
  buildFeatureDetails,
  type FeatureCredit,
  type FeatureDetails,
  type FeatureDetailsHero,
} from "./featureChromeModel";
import { FeatureCompareView } from "./FeatureCompareView";
import { resolveFeatureDetailsHero } from "./featureDetailsHero";
import { featureDetailsHeroPlayerProps } from "./featureDetailsHeroPlayer";
import styles from "./FeatureDetailsModal.module.scss";

export const FEATURE_DETAILS_MODAL_NAME = "feature_details";

export function FeatureDetailsModal({
  featureId,
}: {
  featureId: HomeFeatureId;
}) {
  const details = buildFeatureDetails(featureId);
  const { hideModal } = useModalContext();
  const hero = resolveFeatureDetailsHero(featureId);

  return (
    <div className={styles.split}>
      <section className={styles.info}>
        <FeatureDetailsBody details={details} />
      </section>
      <aside className={styles.hero}>
        <FeatureDetailsHeroMedia hero={hero} title={details.title} />
        <RoundCloseButton
          className={styles.close}
          onClick={() => hideModal("close")}
        />
      </aside>
    </div>
  );
}

function FeatureDetailsBody({ details }: { details: FeatureDetails }) {
  const externalLink = details.externalLink;

  return (
    <>
      <div className={styles.summary}>
        <Text as="h2" variant="heading" size="xl">
          {details.title}
        </Text>
        <FeatureCreditLine credit={details.credit} />
        {details.credit.affiliation === "community" ? (
          <Text as="p" variant="body" size="sm" className={styles.disclaimer}>
            {COMMUNITY_FEATURE_DISCLAIMER}
          </Text>
        ) : null}
        <Text as="p" variant="body" size="lg" className={styles.blurb}>
          {details.blurb}
        </Text>
      </div>
      <div className={styles.specBlock}>
        <dl className={styles.specList}>
          {details.rows.map((row) => (
            <div key={row.id}>
              <dt>
                <Text variant="label" size="md">
                  {row.label}
                </Text>
              </dt>
              <dd>
                <Text variant="body" size="md">
                  {row.href ? (
                    <FeatureDetailsExternalLink href={row.href} label={row.value} />
                  ) : (
                    row.value
                  )}
                </Text>
              </dd>
            </div>
          ))}
        </dl>
        {externalLink ? (
          <div className={styles.actions}>
            <Button
              appearance="neutral"
              hierarchy="primary"
              size="lg"
              label={externalLink.label}
              leftIcon={<ExternalLinkIcon />}
              className={styles.actionButton}
              onClick={() => openExternalBrowserUrl(externalLink.url)}
            />
          </div>
        ) : null}
      </div>
    </>
  );
}

function FeatureDetailsExternalLink({
  href,
  label,
}: {
  href: string;
  label: string;
}) {
  const intercept = shouldInterceptHomeExternalUrl();

  const openExternally = (event: MouseEvent<HTMLAnchorElement>) => {
    if (!intercept) return;
    event.preventDefault();
    openExternalBrowserUrl(href);
  };

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={openExternally}
      onAuxClick={(event) => {
        if (event.button !== 1) return;
        openExternally(event);
      }}
    >
      {label}
    </a>
  );
}

function FeatureCreditLine({ credit }: { credit: FeatureCredit }) {
  return (
    <div className={styles.meta}>
      <span className={styles.avatar} aria-hidden="true">
        {credit.affiliation === "ltx" ? (
          <LtxLogo className={styles.avatarMark} aria-hidden="true" />
        ) : (
          <Text variant="label" size="md">
            C
          </Text>
        )}
      </span>
      <Text variant="label" size="md">
        {`By ${credit.name}`}
      </Text>
    </div>
  );
}

function FeatureDetailsHeroMedia({
  hero,
  title,
}: {
  hero: FeatureDetailsHero;
  title: string;
}) {
  const reduceMotion = useReducedMotion();
  if (hero.compare != null) {
    return (
      <div className={styles.heroPlayer}>
        <FeatureCompareView compare={hero.compare} title={title} />
      </div>
    );
  }

  const videoUrl = hero.videoUrl;
  if (videoUrl == null) {
    return (
      <img
        className={styles.heroMedia}
        src={hero.posterUrl}
        alt={title}
      />
    );
  }

  return (
    <div className={styles.heroPlayer}>
      <VideoPlayer
        {...featureDetailsHeroPlayerProps(
          { videoUrl, posterUrl: hero.posterUrl },
          title,
          reduceMotion,
        )}
      />
    </div>
  );
}
