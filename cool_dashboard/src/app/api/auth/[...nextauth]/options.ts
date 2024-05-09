import type { NextAuthOptions } from "next-auth";
import { prisma } from "@/lib/prisma";
import { compare } from "bcrypt";
import CredentialsProvider from "next-auth/providers/credentials";
import { SupabaseAdapter } from "@auth/supabase-adapter";
export const options: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: {
          label: "Email",
          type: "email",
          placeholder: "hello@example.com",
        },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials): Promise<any> {
        if (!credentials?.email || !credentials.password) {
          return null;
        }

        const user = await prisma.user.findUnique({
          where: {
            email: credentials.email,
          },
        });

        if (!user) {
          return null;
        }

        const isPasswordValid =
          /* credentials.password == user.password; */
          await compare(credentials.password, user.password);

        if (!isPasswordValid) {
          return null;
        }

        return {
          id: user.id + "",
          email: user.email,
          userName: user.userName,
          randomKey: "Hey cool",
          role: user.role,
        };
        /* return await signInWithEmailAndPassword(
            auth,
            (credentials as any).email || "",
            (credentials as any).password || ""
          )
            .then(async (responseFromFirebase) => {
            
              const user = await auth.currentUser;
              return user;
            })
 */
        /* if (
          (credentials as any).email == "test1@mail.com" &&
          (credentials as any).password == "test1234"
        ) {
          const user = {
            id: "testId",
            name: "testName",
            email: (credentials as any).email,
            role: "admin",
          };
          return user;
        } */

        /* .catch((error) => console.log(error))
            .catch((error) => {
              const errorCode = error.code;
              const errorMessage = error.message;
              console.log(error);
            }); */
      },
    }),
  ],
  pages: {
    signIn: "/auth/login",
  },
  callbacks: {
    async jwt({ token, user }) {
      console.log("JWT Callback", { token, user });
      if (user) {
        const u = user as unknown as any;
        return {
          ...token,
          id: u.id,
          randomKey: u.randomKey,
          userName: u.userName,
          role: u.role,
        };
      }
      return token;
    },
    async session({ session, token }) {
      console.log("Session Callback", { session, token });
      return {
        ...session,
        user: {
          ...session.user,
          id: token.id as string,
          randomKey: token.randomKey as string,
          role: token.role as string,
          userName: token.userName as string,
          email: token.email as string,
        },
      };
    },

    async redirect({ url, baseUrl }) {
      // Allows relative callback URLs
      if (url.startsWith("/")) return `${baseUrl}${url}`;
      // Allows callback URLs on the same origin
      else if (new URL(url).origin === baseUrl) return url;
      return baseUrl;
    },
  },
  /* adapter: SupabaseAdapter({
    url: process.env.SUPABASE_URL ?? "",
    secret: process.env.SUPABASE_SERVICE_ROLE_KEY ?? "",
  }), */

  session: {
    strategy: "jwt",
  },
  secret: process.env.NEXTAUTH_SECRET,
  debug: process.env.NODE_ENV === "development",
};
