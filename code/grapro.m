fid = fopen('b.txt','wt');
x = [1,2,3,4];
fprintf(fid,'1 %f\n',var(x));
x = [1,2,3,4];
fprintf(fid,'10 %f\n',var(x));
fclose(fid);
