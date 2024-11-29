import React from "react";
import ReactDOM from "react-dom";
import "./index.css"; // Optional: Include default styles
import App from "./app"; // Import the main App component
import reportWebVitals from "./reportWebVitals"; // Optional: For performance monitoring

// Render the App component to the root element in index.html
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
);

// Optional: Measure app performance (can be removed if not needed)
reportWebVitals();
