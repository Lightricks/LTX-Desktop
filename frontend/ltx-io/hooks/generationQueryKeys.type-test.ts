import { createDesktopExploreRuntime } from "../runtime/createDesktopExploreRuntime.ts";
import type { ExploreModelsVersion } from "../runtime/ExploreRuntime.ts";
import { createRemoteExploreRuntime } from "../../remote/remote-explore-runtime.ts";
import { generationQueryKeys } from "./generationQueryKeys.ts";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Expect<T extends true> = T;

const desktopVersionAllowed: Expect<
  number extends ExploreModelsVersion ? true : false
> = true;
const remoteVersionIsNull: Expect<
  null extends ExploreModelsVersion ? true : false
> = true;
const stringVersionForbidden: Expect<
  string extends ExploreModelsVersion ? false : true
> = true;

const desktopSpecsKey = generationQueryKeys.specs(3);
const remoteSpecsKey = generationQueryKeys.specs(null);

const desktopSpecsKeyType: Expect<
  Equal<typeof desktopSpecsKey, readonly ["video-generation-model-specs", 3]>
> = true;
const remoteSpecsKeyType: Expect<
  Equal<typeof remoteSpecsKey, readonly ["video-generation-model-specs", null]>
> = true;

const desktopRuntimeVersion: Expect<
  Equal<
    ReturnType<typeof createDesktopExploreRuntime>["modelsVersion"],
    number
  >
> = true;
const remoteRuntimeVersion: Expect<
  Equal<ReturnType<typeof createRemoteExploreRuntime>["modelsVersion"], null>
> = true;

// @ts-expect-error Remote has no Desktop modelsVersion argument.
createRemoteExploreRuntime(0);

void [
  desktopVersionAllowed,
  remoteVersionIsNull,
  stringVersionForbidden,
  desktopSpecsKeyType,
  remoteSpecsKeyType,
  desktopRuntimeVersion,
  remoteRuntimeVersion,
];
