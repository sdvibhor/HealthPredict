const CACHE_NAME = 'healthpredict-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/css/style.css',
  '/manifest.json',
  '/static/images/landing.jpg', // Main landing image
  '/static/images/icon-192.png',
  '/static/images/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            return caches.delete(cache);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version if found
        if (response) {
          return response;
        }

        // Otherwise fetch from network
        return fetch(event.request)
          .then((networkResponse) => {
            // Cache new requests dynamically (optional, uncomment to enable)
            // if (event.request.url.startsWith(self.location.origin)) {
            //   const responseClone = networkResponse.clone();
            //   caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseClone));
            // }
            return networkResponse;
          })
          .catch(() => {
            // Offline fallback message if navigating
            if (event.request.mode === 'navigate') {
              return caches.match('/');
            }
          });
      })
  );
});
