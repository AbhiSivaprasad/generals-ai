import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './replays';

import * as serviceWorker from './serviceWorker';

const rootElement = document.getElementById('root')
if (!rootElement) throw new Error('Failed to find the root element');
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

serviceWorker.unregister();
