"use server";
import { User } from "@prisma/client";
import { prisma } from "../../lib/prisma";
import { hash } from "bcrypt";
import { getServerSession } from "next-auth/next";
import { options } from "../../app/api/auth/[...nextauth]/options";
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

      console.log(userName, email, password);
      const user = await prisma.user.create({
        data: {
          userName,
          email,
          password,
          role: "admin", // Note: You should hash the password before storing it
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
export async function sessionGetter() {
  try {
    const session = await getServerSession(options);
    console.log("server action");
    console.log(session?.user);
    console.log("server action");
    return session?.user;
  } catch (error) {
    return { error };
  }
}
export async function returnAnomalyEvent(edgeName: string, groupName: string) {
  try {
    const anomalyEvent = await prisma.anomalyEvent.findMany({
      where: {
        AND: [{ edgeName: edgeName }, { groupName: groupName }],
      },
    });
    return anomalyEvent;
  } catch (error) {
    return { error };
  }
}
export async function returnEdgeGroup() {
  try {
    const edgeGroup = await prisma.edgeGroup.findMany({});
    return edgeGroup;
  } catch (error) {
    return { error };
  }
}
export async function returnEdgeDevice() {
  try {
    const edgeDevice = await prisma.edgeDevice.findMany({});
    return edgeDevice;
  } catch (error) {
    return { error };
  }
}
