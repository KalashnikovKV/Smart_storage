import { NavLink, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>ML Studio</h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Разметка
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Обзор
          </NavLink>
          <NavLink to="/dataset" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Датасет
          </NavLink>
          <NavLink to="/train" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            Обучение
          </NavLink>
        </nav>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
