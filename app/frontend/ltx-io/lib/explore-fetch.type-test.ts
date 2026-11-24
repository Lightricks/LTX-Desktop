import type { components } from "../../generated/backend-openapi.ts";
import type {
  ApiErrors,
  ApiSuccess,
  BoundApiClient,
} from "../../lib/api-client.ts";
import type { ExploreApi } from "../runtime/ExploreRuntime.ts";
import { fetchAsset } from "../hooks/generationQueryKeys.ts";
import { fetchGenerations } from "../hooks/generationQueryKeys.ts";
import type {
  ExploreAsset,
  ExploreAssetListResponse,
  ExploreGeneration,
  ExploreListedAsset,
} from "../../lib/explore-contract.ts";
import type { Generation } from "./resultsFeedModel.ts";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Expect<T extends true> = T;

type DesktopAsset = components["schemas"]["Asset"];
type DesktopAssetListItem = components["schemas"]["AssetListItem"];
type DesktopGeneration = components["schemas"]["Generation"];
type HTTPErrorResponse = components["schemas"]["HTTPErrorResponse"];

type ExploreListResult = Awaited<ReturnType<ExploreApi["listAssets"]>>;
type ExploreListSuccess = ApiSuccess<ExploreListResult>;
type ExploreListItem = ExploreListSuccess["items"][number];
type ExploreListQuery = Parameters<ExploreApi["listAssets"]>[0];

const fetchAssetReturnsExplore: Expect<
  Equal<Awaited<ReturnType<typeof fetchAsset>>, ExploreAsset>
> = true;
const fetchGenerationsReturnExplore: Expect<
  Equal<Awaited<ReturnType<typeof fetchGenerations>>, ExploreGeneration[]>
> = true;
const resultFeedUsesExploreGeneration: Expect<
  Equal<Generation, ExploreGeneration>
> = true;
const desktopAssetFitsFetch: Expect<
  DesktopAsset extends Awaited<ReturnType<typeof fetchAsset>> ? true : false
> = true;
const desktopGenerationFitsList: Expect<
  DesktopGeneration extends Awaited<
    ReturnType<typeof fetchGenerations>
  >[number]
    ? true
    : false
> = true;

const fetchAssetTakesExploreApi: Expect<
  Equal<Parameters<typeof fetchAsset>[0], ExploreApi>
> = true;
const fetchGenerationsTakesExploreApi: Expect<
  Equal<Parameters<typeof fetchGenerations>[0], ExploreApi>
> = true;

const boundClientFitsExploreApi: Expect<
  BoundApiClient extends ExploreApi ? true : false
> = true;
const listItemsAreExploreListed: Expect<
  Equal<ExploreListItem, ExploreListedAsset>
> = true;
const listSuccessIsExploreListResponse: Expect<
  Equal<ExploreListSuccess, ExploreAssetListResponse>
> = true;
const listedPathIsNotRequired: Expect<
  ExploreListItem extends { path: string } ? false : true
> = true;
const listedRequiresInUse: Expect<
  ExploreListItem extends { in_use: boolean } ? true : false
> = true;
const listedThumbnailFlagIsOptional: Expect<
  ExploreListItem extends { has_thumbnail: boolean } ? false : true
> = true;
const desktopListItemFitsExplore: Expect<
  DesktopAssetListItem extends ExploreListItem ? true : false
> = true;
const listQueryPreserved: Expect<
  Equal<
    ExploreListQuery,
    {
      media_kind?: ("image" | "video" | "audio") | null;
      sort?: "created_at-desc" | "created_at-asc";
      q?: string | null;
      cursor?: string | null;
      limit?: number;
    }
  >
> = true;
const listErrorsAreStandardHttp: Expect<
  Equal<
    ApiErrors<ExploreListResult>,
    | { status: "4XX"; error: HTTPErrorResponse }
    | { status: "5XX"; error: HTTPErrorResponse }
    | { status: "default"; error: HTTPErrorResponse }
  >
> = true;

const pathFreeRemoteListedAsset: ExploreListItem = {
  created_at: 1,
  id: "asset-1",
  in_use: false,
  has_thumbnail: true,
  media_kind: "image",
  metadata: {
    mediaType: "image",
    metadata: { width: 8, height: 8 },
  },
  mime_type: "image/png",
  name: "photo.png",
  origin: "generated",
};
const pathFreeListSuccess: ExploreListResult = {
  ok: true,
  data: {
    items: [pathFreeRemoteListedAsset],
    next_cursor: null,
  },
};
const desktopHostFieldsOnListed: ExploreListItem = {
  ...pathFreeRemoteListedAsset,
  path: "/tmp/photo.png",
  thumbnail_path: "/tmp/photo-thumb.png",
};

void [
  fetchAssetReturnsExplore,
  fetchGenerationsReturnExplore,
  resultFeedUsesExploreGeneration,
  desktopAssetFitsFetch,
  desktopGenerationFitsList,
  fetchAssetTakesExploreApi,
  fetchGenerationsTakesExploreApi,
  boundClientFitsExploreApi,
  listItemsAreExploreListed,
  listSuccessIsExploreListResponse,
  listedPathIsNotRequired,
  listedRequiresInUse,
  listedThumbnailFlagIsOptional,
  desktopListItemFitsExplore,
  listQueryPreserved,
  listErrorsAreStandardHttp,
  pathFreeRemoteListedAsset,
  pathFreeListSuccess,
  desktopHostFieldsOnListed,
];
