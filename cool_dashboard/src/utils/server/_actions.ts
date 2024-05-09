"use server";
import { User } from "@prisma/client";
import { prisma } from "../../lib/prisma";
import { hash } from "bcrypt";

export async function createUser(
  userName: string,
  email: string,
  password: string,
) {
  try {
    const exists = await userExists(email);
    if (exists) {
      return { error: "User already exists" };
    } else {
      var hashedPassword = (await hash(password, 12)) as string;
      password = hashedPassword;
      console.log("server action");
      console.log(userName, email, password);
      const user = await prisma.user.create({
        data: {
          userName,
          email,
          password, // Note: You should hash the password before storing it
        },
      });
      console.log("User created:", user);
      /* return { user }; */
    }
    // Create user\
  } catch (error) {
    return false;
  }
}
export async function findUser(email: string) {
  try {
    const user = await prisma.user.findUnique({
      where: {
        email: email,
      },
    });
    return user;
  } catch (error) {
    return { error };
  }
}
export async function userExists(email: string) {
  try {
    const exists = await findUser(email);
    if (exists) {
      return true;
    } else {
      return false;
    }
  } catch (error) {
    return { error };
  }
}
