import { lazy, Suspense, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./App.css";

// Hash routes keep this a plain static site (no server rewrites needed). Pages are lazy so the
// small /#/faces page doesn't download Cesium.
const App = lazy(() => import("./App.jsx"));
const FacesPage = lazy(() => import("./FacesPage.jsx"));

const subscribe = (cb) => {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
};
const getRoute = () => window.location.hash.replace(/^#/, "") || "/";

function Root() {
  const route = useSyncExternalStore(subscribe, getRoute);
  return <Suspense fallback={null}>{route === "/faces" ? <FacesPage /> : <App />}</Suspense>;
}

createRoot(document.getElementById("root")).render(<Root />);
