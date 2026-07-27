import { createRouter } from '@tanstack/react-router'
import { routeTree } from './routeTree.gen'

/** TanStack Start calls getRouter() on the server for each request. */
export function getRouter() {
  return createRouter({
    routeTree,
    defaultPreload: 'intent',
    scrollRestoration: true,
  })
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
