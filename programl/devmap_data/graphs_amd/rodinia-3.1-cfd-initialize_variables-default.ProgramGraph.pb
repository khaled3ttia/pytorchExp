

[external]
KcallBC
A
	full_text4
2
0%4 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
3icmpB+
)
	full_text

%6 = icmp slt i32 %5, %2
"i32B

	full_text


i32 %5
6brB0
.
	full_text!

br i1 %6, label %7, label %42
 i1B

	full_text	

i1 %6
4sext8B*
(
	full_text

%8 = sext i32 %2 to i64
/shl8B&
$
	full_text

%9 = shl i64 %4, 32
$i648B

	full_text


i64 %4
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
?bitcast8B2
0
	full_text#
!
%11 = bitcast float* %1 to i32*
Hload8B>
<
	full_text/
-
+%12 = load i32, i32* %11, align 4, !tbaa !8
'i32*8B

	full_text


i32* %11
\getelementptr8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
@bitcast8B3
1
	full_text$
"
 %14 = bitcast float* %13 to i32*
+float*8B

	full_text


float* %13
Hstore8B=
;
	full_text.
,
*store i32 %12, i32* %14, align 4, !tbaa !8
%i328B

	full_text
	
i32 %12
'i32*8B

	full_text


i32* %14
Zgetelementptr8BG
E
	full_text8
6
4%15 = getelementptr inbounds float, float* %1, i64 1
@bitcast8B3
1
	full_text$
"
 %16 = bitcast float* %15 to i32*
+float*8B

	full_text


float* %15
Hload8B>
<
	full_text/
-
+%17 = load i32, i32* %16, align 4, !tbaa !8
'i32*8B

	full_text


i32* %16
5add8B,
*
	full_text

%18 = add nsw i64 %10, %8
%i648B

	full_text
	
i64 %10
$i648B

	full_text


i64 %8
\getelementptr8BI
G
	full_text:
8
6%19 = getelementptr inbounds float, float* %0, i64 %18
%i648B

	full_text
	
i64 %18
@bitcast8B3
1
	full_text$
"
 %20 = bitcast float* %19 to i32*
+float*8B

	full_text


float* %19
Hstore8B=
;
	full_text.
,
*store i32 %17, i32* %20, align 4, !tbaa !8
%i328B

	full_text
	
i32 %17
'i32*8B

	full_text


i32* %20
Zgetelementptr8BG
E
	full_text8
6
4%21 = getelementptr inbounds float, float* %1, i64 2
@bitcast8B3
1
	full_text$
"
 %22 = bitcast float* %21 to i32*
+float*8B

	full_text


float* %21
Hload8B>
<
	full_text/
-
+%23 = load i32, i32* %22, align 4, !tbaa !8
'i32*8B

	full_text


i32* %22
3shl8B*
(
	full_text

%24 = shl nsw i64 %8, 1
$i648B

	full_text


i64 %8
6add8B-
+
	full_text

%25 = add nsw i64 %24, %10
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %0, i64 %25
%i648B

	full_text
	
i64 %25
@bitcast8B3
1
	full_text$
"
 %27 = bitcast float* %26 to i32*
+float*8B

	full_text


float* %26
Hstore8B=
;
	full_text.
,
*store i32 %23, i32* %27, align 4, !tbaa !8
%i328B

	full_text
	
i32 %23
'i32*8B

	full_text


i32* %27
Zgetelementptr8BG
E
	full_text8
6
4%28 = getelementptr inbounds float, float* %1, i64 3
@bitcast8B3
1
	full_text$
"
 %29 = bitcast float* %28 to i32*
+float*8B

	full_text


float* %28
Hload8B>
<
	full_text/
-
+%30 = load i32, i32* %29, align 4, !tbaa !8
'i32*8B

	full_text


i32* %29
3mul8B*
(
	full_text

%31 = mul nsw i64 %8, 3
$i648B

	full_text


i64 %8
6add8B-
+
	full_text

%32 = add nsw i64 %31, %10
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %0, i64 %32
%i648B

	full_text
	
i64 %32
@bitcast8B3
1
	full_text$
"
 %34 = bitcast float* %33 to i32*
+float*8B

	full_text


float* %33
Hstore8B=
;
	full_text.
,
*store i32 %30, i32* %34, align 4, !tbaa !8
%i328B

	full_text
	
i32 %30
'i32*8B

	full_text


i32* %34
Zgetelementptr8BG
E
	full_text8
6
4%35 = getelementptr inbounds float, float* %1, i64 4
@bitcast8B3
1
	full_text$
"
 %36 = bitcast float* %35 to i32*
+float*8B

	full_text


float* %35
Hload8B>
<
	full_text/
-
+%37 = load i32, i32* %36, align 4, !tbaa !8
'i32*8B

	full_text


i32* %36
3shl8B*
(
	full_text

%38 = shl nsw i64 %8, 2
$i648B

	full_text


i64 %8
6add8B-
+
	full_text

%39 = add nsw i64 %38, %10
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %0, i64 %39
%i648B

	full_text
	
i64 %39
@bitcast8B3
1
	full_text$
"
 %41 = bitcast float* %40 to i32*
+float*8B

	full_text


float* %40
Hstore8B=
;
	full_text.
,
*store i32 %37, i32* %41, align 4, !tbaa !8
%i328B

	full_text
	
i32 %37
'i32*8B

	full_text


i32* %41
'br8B

	full_text

br label %42
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1       	
 		                        !" !! #$ #% ## && '( '' )* )) +, ++ -. -/ -- 01 00 23 22 45 46 44 77 89 88 :; :: <= << >? >@ >> AB AA CD CC EF EG EE HH IJ II KL KK MN MM OP OQ OO RS RR TU TT VW VX VV Y[ [ \ \ \ 0\ A\ R] ] ] &] 7] H    
	             " $! %& (' * ,+ . /- 10 3) 52 67 98 ; =< ? @> BA D: FC GH JI L NM P QO SR UK WT X  ZY Z Z ^^ ^^ _ H` 	` a &a Mb 7b <c d d +"
initialize_variables"
_Z13get_global_idj*?
'rodinia-3.1-cfd-initialize_variables.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
I??A

wgsize_log1p
I??A

transfer_bytes
???

devmap_label
 

wgsize
?