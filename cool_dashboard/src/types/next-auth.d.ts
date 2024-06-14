import { DefaultSession } from "next-auth";
declare module "next-auth" {
  interface Session {
    user?: {
      name?: string | null | undefined;
      email?: string | null | undefined;
      image?: string | null | undefined;
      // Add your new attribute here
      role?: string | null | undefined; // For example, adding a role attribute
    }; // Ensure the default properties are still included
  }
}
