import { prisma } from "@/lib/prisma";

export default async function saveAnomaly(message: any) {
  console.log(message.msg);
  await prisma.user.create({
    data: {
      userName: "Alice234",
      email: "alice@prisma.io234",
      password: message.msg,
      role: "admin",
    },
  });
  const allUsers = await prisma.user.findMany();
  /*   console.log(allUsers); */
  return allUsers;
}
