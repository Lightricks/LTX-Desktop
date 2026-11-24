import { isManagedModelKind } from "@/lib/managed-model-setup";
import { useState, type ReactNode } from "react";
import { Button } from "@ds/Button/Button";
import { PickerModelSetup } from "./PickerModelSetup";
import { Link } from "react-router";

import formStyles from "../FeatureFormView.module.scss";
import type {
  FeatureFieldSchema,
  FieldOption,
  OptionsFieldDefForKey,
  OptionsFieldKey,
  ValidationIssue,
} from "../types";
import { FieldLabel } from "./FieldLabel";
import { FormOptionControl } from "./FormOptionControl";

export function SettingField<
  TSchema extends FeatureFieldSchema,
  TKey extends OptionsFieldKey<TSchema>,
>({
  field,
  value,
  issue,
  onChange,
  getOptionIcon,
}: {
  field: OptionsFieldDefForKey<TSchema, TKey>;
  value: Extract<TSchema[TKey]["value"], string | number>;
  issue?: ValidationIssue;
  onChange: (value: Extract<TSchema[TKey]["value"], string | number>) => void;
  getOptionIcon?: (
    option: FieldOption<Extract<TSchema[TKey]["value"], string | number>>,
  ) => ReactNode;
}) {
  type Value = Extract<TSchema[TKey]["value"], string | number>;
  const resumeKey = field.id === 'model' && typeof window !== 'undefined' && window.electronAPI
    ? `ltx.amd.video-picker-setup.v1.${window.location.pathname}${window.location.hash}${window.location.search}` : null;
  const [pending, setPending] = useState<{ value: Value; setup: NonNullable<FieldOption["setup"]> } | null>(() => {
    const kind = resumeKey ? window.localStorage.getItem(resumeKey) : null;
    if (!isManagedModelKind(kind)) return null;
    return { value: (kind === 'comfy-int8' ? 'comfy' : kind) as Value, setup: { kind, label: 'Set up' } };
  });
  const selected = field.options.find(option => option.value === value);
  const requestSetup = (option: FieldOption<Value>) => {
    if (option.setup) {
      if (resumeKey && isManagedModelKind(option.setup.kind)) {
        window.localStorage.setItem(resumeKey, option.setup.kind);
      } else if (resumeKey) window.localStorage.removeItem(resumeKey);
      setPending({ value: option.value, setup: option.setup });
    }
  };
  const closeSetup = () => {
    if (resumeKey) window.localStorage.removeItem(resumeKey);
    setPending(null);
  };
  const select = (next: Value) => {
    closeSetup();
    onChange(next);
  };

  return (
    <div className={formStyles.field}>
      <FieldLabel label={field.label} tooltip={field.tooltip} />
      <div className={formStyles.fieldControl}>
        <FormOptionControl
          options={field.options}
          value={value}
          onChange={select}
          onRequestSetup={requestSetup}
          ariaLabel={field.label}
          forcePicker={field.forcePicker}
          maxOptionCount={field.maxOptionCount}
          getOptionIcon={getOptionIcon}
          showListIcons={field.dataKey === "aspectRatio"}
        />
      </div>
      {pending ? (
        <PickerModelSetup
          key={pending.setup.kind}
          kind={pending.setup.kind}
          option={field.options.find(option => option.value === pending.value)}
          onUse={select}
          onClose={closeSetup}
        />
      ) : selected?.setup ? (
        <Button appearance="neutral" hierarchy="secondary" size="sm"
          label={selected.setup.label}
          onClick={() => requestSetup(selected)}
        />
      ) : null}
      {field.relatedPage ? <Link to={field.relatedPage.to} className={formStyles.fieldNotice}>{field.relatedPage.label}</Link> : null}
      {issue ? (
        <span className={formStyles.fieldError}>{issue.message}</span>
      ) : null}
    </div>
  );
}
