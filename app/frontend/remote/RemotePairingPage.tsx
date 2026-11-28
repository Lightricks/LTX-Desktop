import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { tokenFromLocation } from "./api.ts";
import { completeRemotePairing } from "./completeRemotePairing.ts";
import { PairingStatus, type PairingHealth } from "./PairingMessage.tsx";

export function RemotePairingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [health, setHealth] = useState<Exclude<PairingHealth, "ok">>("checking");
  const grant = tokenFromLocation(location);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const paired = await completeRemotePairing(grant, navigate);
      if (cancelled) {
        return;
      }
      if (paired !== "ok") {
        setHealth(paired);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [grant, navigate]);

  return <PairingStatus health={health} />;
}
