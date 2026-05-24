import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { DatasetPage } from "./pages/DatasetPage";
import { LabelPage } from "./pages/LabelPage";
import { TrainPage } from "./pages/TrainPage";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<LabelPage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="dataset" element={<DatasetPage />} />
          <Route path="train" element={<TrainPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
