

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
2icmpB*
(
	full_text

%5 = icmp sgt i32 %1, 0
5brB/
-
	full_text 

br i1 %5, label %6, label %8
 i1B

	full_text	

i1 %5
4sext8B*
(
	full_text

%7 = sext i32 %1 to i64
&br8B

	full_text

br label %9
$ret8B

	full_text


ret void
@phi8B7
5
	full_text(
&
$%10 = phi i64 [ 0, %6 ], [ %20, %9 ]
%i648B

	full_text
	
i64 %20
8trunc8B-
+
	full_text

%11 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
1shl8B(
&
	full_text

%12 = shl i32 %11, 12
%i328B

	full_text
	
i32 %11
5add8B,
*
	full_text

%13 = add nsw i32 %12, %4
%i328B

	full_text
	
i32 %12
$i328B

	full_text


i32 %4
6sext8B,
*
	full_text

%14 = sext i32 %13 to i64
%i328B

	full_text
	
i32 %13
\getelementptr8BI
G
	full_text:
8
6%15 = getelementptr inbounds float, float* %0, i64 %14
%i648B

	full_text
	
i64 %14
1or8B)
'
	full_text

%16 = or i32 %12, 4096
%i328B

	full_text
	
i32 %12
5add8B,
*
	full_text

%17 = add nsw i32 %16, %4
%i328B

	full_text
	
i32 %16
$i328B

	full_text


i32 %4
6sext8B,
*
	full_text

%18 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
\getelementptr8BI
G
	full_text:
8
6%19 = getelementptr inbounds float, float* %0, i64 %18
%i648B

	full_text
	
i64 %18
Vcall8BL
J
	full_text=
;
9tail call void @BoxMullerTrans(float* %15, float* %19) #4
+float*8B

	full_text


float* %15
+float*8B

	full_text


float* %19
4add8B+
)
	full_text

%20 = add nuw i64 %10, 2
%i648B

	full_text
	
i64 %10
7icmp8B-
+
	full_text

%21 = icmp slt i64 %20, %7
%i648B

	full_text
	
i64 %20
$i648B

	full_text


i64 %7
8br8B0
.
	full_text!

br i1 %21, label %9, label %8
#i18B

	full_text


i1 %21
$i328B

	full_text


i32 %1
*float*8B

	full_text

	float* %0
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 0
&i328B

	full_text


i32 4096
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 12       

                      !  "    #$ ## %& %' %% () (* * + +   # 
            ! "
 $# & '% )  	 
( 
( 	 	 -- ,, ,,   --  . 
/ 0 #1 1 2 "
	BoxMuller"
_Z13get_global_idj"
BoxMullerTrans*?
'nvidia-4.2-MersenneTwister-BoxMuller.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?

wgsize_log1p
"??A

transfer_bytes
???[

devmap_label

 
transfer_bytes_log1p
"??A