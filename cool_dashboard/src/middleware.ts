export { default } from "next-auth/middleware";

export const config = {
  matcher: [
    "/ui-components/forms",
    "/ui-components/clientComponent",
    "/ui-components/serverComponent",
    "/ui-components/table",
    "/ui-components/images",
    "/ui-components/alerts",
    "/ui-components/pagination",
    "/ui-components/buttons",
    "/",
  ],
};
