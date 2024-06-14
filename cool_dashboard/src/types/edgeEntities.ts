// types/index.d.ts
import {
  AnomalyEvent as PrismaAnomalyEvent,
  EdgeDevice as PrismaEdgeDevice,
  EdgeGroup as PrismaEdgeGroup,
  User as PrismaUser,
} from "@prisma/client";

export type AnomalyEvent = PrismaAnomalyEvent[] | ErrorResponse;
export type EdgeDevice = PrismaEdgeDevice[] | ErrorResponse;
export type EdgeGroup = PrismaEdgeGroup[] | ErrorResponse;
export type User = PrismaUser[] | ErrorResponse;
export interface ErrorResponse {
  error: string;
}
export type Result<T> = T | ErrorResponse;
