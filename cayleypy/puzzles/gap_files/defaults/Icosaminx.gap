# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap Icosaminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_FREGU:=(5,20,43,17,50)(6,19,44,18,49)(86,90,89,88,87)(121,122,128,129,123);
M_M_OKBDN:=(31,37,36,39,34)(32,38,35,40,33)(96,100,99,98,97)(132,134,140,135,133);
M_M_HIERC:=(1,6,25,30,27)(2,5,26,29,28)(61,65,64,63,62)(121,123,131,125,124);
M_M_MKOPQ:=(23,40,55,51,42)(24,39,56,52,41)(111,115,114,113,112)(130,135,140,139,136);
M_M_FLACR:=(1,8,14,3,20)(2,7,13,4,19)(66,70,69,68,67)(121,124,126,127,122);
M_M_BKMJS:=(35,60,53,47,56)(36,59,54,48,55)(101,105,104,103,102)(134,138,137,139,140);
M_M_NALPO:=(11,34,23,22,13)(12,33,24,21,14)(91,95,94,93,92)(126,133,135,130,127);
M_M_JGEIS:=(25,50,45,53,58)(26,49,46,54,57)(116,120,119,118,117)(123,129,137,138,131);
M_M_FUQPL:=(3,22,41,16,44)(4,21,42,15,43)(81,85,84,83,82)(122,127,130,136,128);
M_M_IHDBS:=(9,30,57,60,37)(10,29,58,59,38)(76,80,79,78,77)(125,131,138,134,132);
M_M_UGJMQ:=(15,52,48,45,18)(16,51,47,46,17)(106,110,109,108,107)(128,136,139,137,129);
M_M_DHCAN:=(7,27,10,31,12)(8,28,9,32,11)(71,75,74,73,72)(124,125,132,133,126);
Gen:=[
M_M_FREGU,M_M_OKBDN,M_M_HIERC,M_M_MKOPQ,M_M_FLACR,M_M_BKMJS,M_M_NALPO,M_M_JGEIS,M_M_FUQPL,M_M_IHDBS,M_M_UGJMQ,M_M_DHCAN
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25],[27],[29],[31],[33],[35],[37],[39],[41],[43],[45],[47],[49],[51],[53],[55],[57],[59],[61],[66],[71],[76],[81],[86],[91],[96],[101],[106],[111],[116],[121],[122],[123],[124],[125],[126],[127],[128],[129],[130],[131],[132],[133],[134],[135],[136],[137],[138],[139],[140]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

